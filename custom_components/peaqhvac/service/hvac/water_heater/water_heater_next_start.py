from statistics import mean, stdev
from datetime import datetime, timedelta
import logging

from enum import Enum


class GroupType(Enum):
    LOW = "cheap"
    MID = "medium"
    HIGH = "expensive"
    UNKNOWN = "unknown"
    FLAT = "flat"


class Demand(Enum):
    NoDemand = "No demand"
    LowDemand = "Low demand"
    MediumDemand = "Medium demand"
    HighDemand = "High demand"


class HvacPresets(Enum):
    Normal = 1
    Eco = 2
    Away = 3
    ExtendedAway = 4


class Group:
    def __init__(self, group_type: GroupType, hours: list[int]):
        self.group_type = group_type
        self.hours = hours


_LOGGER = logging.getLogger(__name__)

HOUR_LIMIT = 18
DELAY_LIMIT = 48
MIN_DEMAND = 26
DEFAULT_TEMP_TREND = -0.4

DEMAND_MINUTES = {
    HvacPresets.Normal: {
        Demand.NoDemand:     0,
        Demand.LowDemand:    26,
        Demand.MediumDemand: 35,
        Demand.HighDemand:   45
    },
    HvacPresets.Eco:    {
        Demand.NoDemand:     0,
        Demand.LowDemand:    26,
        Demand.MediumDemand: 35,
        Demand.HighDemand:   45
    },
    HvacPresets.Away:   {
        Demand.NoDemand:     0,
        Demand.LowDemand:    0,
        Demand.MediumDemand: 26,
        Demand.HighDemand:   26
    }
}


def get_demand(temp) -> Demand:
    if temp is None:
        return Demand.NoDemand
    if 0 < temp < 100:
        if temp >= 40:
            return Demand.NoDemand
        if temp > 35:
            return Demand.LowDemand
        if temp >= 25:
            return Demand.MediumDemand
        if temp < 25:
            return Demand.HighDemand
    return Demand.NoDemand


DEMAND_HOURS = [6, 20, 21, 22, 18, 19]
NON_HOURS = [7, 8, 11, 12, 16, 17]


class NextStartParams:
    delay_hours: int
    is_cold: bool
    cold_target_dt: datetime
    last_known_price: datetime
    cold_before_last_known_price: bool

    def __init__(self, now_dt, prices, temp, target_temp, temp_trend):
        self.delay_hours = NextStartParams.get_hours_til_cold(
            target_temp,
            temp,
            temp_trend
        )
        self.cold_target_dt = self._set_cold_target(now_dt)
        self.is_cold = False if self.delay_hours == 0 else True
        self.last_known_price = self._set_last_known_price(now_dt, prices)
        self.cold_before_last_known_price = self._set_cold_before_last_known_price(
            now_dt,
            self.last_known_price
        )

    def _set_cold_before_last_known_price(self, now_dt, last_known_price):
        return self.delay_hours < int((last_known_price - now_dt).total_seconds() / 3600)
    def _set_last_known_price(self, now_dt, prices):
        return now_dt.replace(hour=0, minute=0, second=0) + timedelta(hours=len(prices) - 1)

    def _set_cold_target(self, now_dt):
        return now_dt if self.delay_hours == 0 else now_dt + timedelta(hours=self.delay_hours)

    @staticmethod
    def get_hours_til_cold(target_temp, temp, temp_trend) -> int:
        try:
            if target_temp - temp > 0:
                delay = 0
            else:
                delay = (target_temp - temp) / temp_trend
        except ZeroDivisionError:
            delay = DELAY_LIMIT
        return delay


class NextWaterBoost:
    def __init__(self):
        self.prices = []
        self.min_price: float = None  # type: ignore
        self.groups = []
        self.non_hours: list = []
        self.demand_hours: list = []
        self.preset: HvacPresets = HvacPresets.Normal
        self.now_dt: datetime = None  # type: ignore
        self.floating_mean: float = None  # type: ignore
        self.temp_trend: float = None  # type: ignore
        self.current_temp: float = None  # type: ignore

    def next_predicted_demand(
            self,
            prices_today: list,
            prices_tomorrow: list,
            min_price: float,
            temp: float,
            temp_trend: float,
            target_temp: float,
            demand: int = MIN_DEMAND,
            preset: HvacPresets = HvacPresets.Normal,
            now_dt=None,
            non_hours=None,
            high_demand_hours=None
    ) -> datetime:
        if len(prices_today) < 1:
            return datetime.max
        self._init_vars(temp, temp_trend, prices_today, prices_tomorrow, preset, min_price, non_hours,
                        high_demand_hours, now_dt)

        return self._get_next_start(
            demand=demand,
            target_temp=target_temp,
            temp=temp
        )

    def _init_vars(self, temp, temp_trend, prices_today: list, prices_tomorrow: list, preset: HvacPresets,
                   min_price: float, non_hours: list = None, high_demand_hours: list = None, now_dt=None) -> None:
        if non_hours is None:
            non_hours = []
        if high_demand_hours is None:
            high_demand_hours = []
        self.min_price = min_price
        self.prices = prices_today + prices_tomorrow
        self.non_hours = non_hours
        self.demand_hours = high_demand_hours
        self._set_now_dt(now_dt)
        self._set_floating_mean()
        self._group_prices(prices_today, prices_tomorrow)
        self.preset = preset
        self.temp_trend = DEFAULT_TEMP_TREND if temp_trend == 0 else temp_trend
        self.current_temp = temp

    def _set_now_dt(self, now_dt=None) -> None:
        self.now_dt = datetime.now() if now_dt is None else now_dt

    def _set_floating_mean(self) -> None:
        self.floating_mean = mean(self.prices[self.now_dt.hour:])

    # ------------------------------------------------------------------------------------------------

    def _get_next_start(self, demand: int, temp: float, target_temp: float) -> datetime:
        """Check when we are cold"""
        model = NextStartParams(self.now_dt, self.prices, temp, target_temp, self.temp_trend)

        """Check if we are about to pass through any high demand hours before then (if so, consider the predicted temp at that hour as well)"""
        passing_high_demand = self.passes_through_hour(self.now_dt, model.cold_target_dt, self.demand_hours)
        """Check if we are about to pass through any non hours before then - we should preheat before then probably?"""
        passing_non_hours = self.passes_through_hour(self.now_dt, model.cold_target_dt, self.non_hours)
        print(passing_high_demand, passing_non_hours)

        match model.is_cold:
            case True:
                # case 1 or 2
                pass
            case False:
                # case 3
                pass
            case _:
                raise ValueError("is_cold is not a bool")

        if model.cold_before_last_known_price:
            delay_hour = int(round(self.now_dt.hour + model.delay_hours, 0))
            print(delay_hour)
            now_group = self._find_group(self.now_dt.hour)
            delay_group = self._find_group(delay_hour)

            print("grouping:", now_group.hours)
            print("delayed_grouping:", delay_group.hours)

        else:
            print("price will run out before it is cold. pre-heat")

        """
        valid cases:
        1. water is cold and it is cheap enought right now
        2. water is cold and it is currently not cheap
        3. water is not cold

        a: between now and desired heating we are passing through high demand hours
        b: between now and desired heating we are passing through non hours
        c: between now and desired heating we are passing through high demand hours and non hours
        d: between now and desired heating we are neither passing through high demand hours nor non hours
        e: known hours are too few to calculate properly
        f: hour_limit (18) is longer than the calculation

        1a: -> get_next_start
        1b: -> wait til after non hours and get_next_start
        1c: -> 
        1d: -> get_next_start
        2a: ->
        2b: ->
        2c: ->
        2d: ->
        2e: ->
        2f: ->
        3a: ->
        3b: ->
        3c: ->
        3d: ->
        3e: ->
        3f: ->
        """

        # ------------------
        if model.last_known_price - self.now_dt > timedelta(hours=HOUR_LIMIT) and not model.is_cold:
            """There are prices for tomorrow, but we are not cold, so we can wait until tomorrow"""
            group = self._find_group(self.now_dt.hour)
            print(group)
            return self._calculate_last_start(group.hours if group.group_type == GroupType.LOW else [])

        # _next_dt = self._calculate_next_start(demand)

        # if cold_target_dt:
        #     if _next_dt < cold_target_dt:
        #         return self._calculate_last_start()
        #     return _next_dt.replace(minute=self._set_minute_start(demand))

        # if self.range_not_in_nonhours(_next_dt):
        #     print("here2", _next_dt)
        #     return _next_dt.replace(minute=self._set_minute_start(demand))
        # return self._get_next_start(demand, delay_dt=_next_dt+timedelta(hours=1), cold=is_cold)

    @staticmethod
    def passes_through_hour(a, b, x: list):
        def passes_through_hour_single(a, b, x: int):
            return a.hour <= x < b.hour or (a.hour > b.hour and not (b.hour < x < a.hour))

        if a == b:
            return a.hour in x
        return all([passes_through_hour_single(a, b, h) for h in x])

    # ------------------------------------------------------------------------------------------------

    def _set_minute_start(self, demand) -> int:
        ret = 60 - int(demand / 2)
        if ret not in range(0, 60):
            _LOGGER.error(f"Minute start not in range: {ret}")
            return 59
        return ret

    def _set_start_dt(self, demand: int, delayed_dt: datetime = None) -> datetime:
        now_dt = self.now_dt if delayed_dt is None else delayed_dt
        print("here3", now_dt)
        return now_dt.replace(minute=self._set_minute_start(demand), second=0, microsecond=0)

    # def _get_low_period(self, override_dt=None) -> int:
    #     dt = self.now_dt if override_dt is None else override_dt
    #     if override_dt is not None:
    #         _start_hour = dt.hour + (int(self.now_dt.day != override_dt.day) * 24)
    #     else:
    #         _start_hour = dt.hour
    #     low_period: int = 0
    #     for i in range(_start_hour, len(self.prices)):
    #         if self.prices[i] > self.floating_mean:
    #             break
    #         if i == dt.hour:
    #             low_period = 60 - dt.minute
    #         else:
    #             low_period += 60
    #     return low_period

    # def _values_are_good(self, i) -> bool:
    #     return all([
    #         self.prices[i] < self.floating_mean or self.prices[i] < self.min_price,
    #         self.prices[i + 1] < self.floating_mean or self.prices[i + 1] < self.min_price,
    #         [i, i + 1, i - 23, i - 24] not in self.non_hours,
    #     ])

    # def _calculate_next_start(self, demand: int) -> datetime:
    #     try:
    #         if self.prices[self.now_dt.hour] < self.floating_mean and not any(
    #             [self.now_dt.hour in self.non_hours,
    #              self.now_dt.hour + 1 in self.non_hours]
    #              ):
    #             """This hour is cheap enough to start"""
    #             low_period = self._get_low_period()
    #             return self._set_start_dt(demand=demand, low_period=low_period)
    #         for i in range(self.now_dt.hour, len(self.prices) - 1):
    #             """Search forward for other hours to start"""
    #             if self._values_are_good(i):
    #                 return self._set_start_dt_params(i)
    #     except Exception as e:
    #         _LOGGER.error(f"Error on getting next start: {e}")
    #         return datetime.max

    # def _calculate_last_start(self, group: list = []) -> datetime:
    #     try:
    #         _param_i = None
    #         _range = range(self.now_dt.hour, min(len(self.prices) - 1, self.now_dt.hour + HOUR_LIMIT))
    #         if len(group) > 1:
    #             _range = range(self.now_dt.hour, max(group))
    #         print("checking B", group)
    #         for i in _range:
    #             if self._values_are_good(i):
    #                 _param_i = i
    #         if _param_i is None:
    #             return self._calculate_last_start_reverse()
    #         return self._set_start_dt_params(_param_i)
    #     except Exception as e:
    #         _LOGGER.error(f"Error on getting last close start: {e}")
    #         return datetime.max

    # def _calculate_last_start_reverse(self):
    #     try:
    #         print("checking C")
    #         for i in reversed(range(self.now_dt.hour,min(len(self.prices) - 1, self.now_dt.hour + HOUR_LIMIT))):
    #             if self._values_are_good(i):
    #                 return self._set_start_dt_params(i)
    #     except Exception as e:
    #         _LOGGER.error(f"Error on getting last start: {e}")
    #         return datetime.max

    # def _set_start_dt_params(self, i: int) -> datetime:
    #     delay = (i - self.now_dt.hour)
    #     delayed_dt = self.now_dt + timedelta(hours=delay)
    #     excepted_temp = self.current_temp + (delay * self.temp_trend)
    #     new_demand = max(DEMAND_MINUTES[self.preset][get_demand(excepted_temp)],DEMAND_MINUTES[self.preset][Demand.LowDemand])
    #     return self._set_start_dt(new_demand, delayed_dt)

    def _group_prices(self, prices_today: list, prices_tomorrow: list) -> None:
        today_len = len(prices_today)
        std_dev = stdev(self.prices)
        average = mean(self.prices)
        if len(prices_tomorrow):
            std_dev_tomorrow = stdev(prices_tomorrow)
            average_tomorrow = mean(prices_tomorrow)
        continuous_groups = []
        current_group = [0]

        def __set_group_type(_average, flat, average):
            if flat:
                return GroupType.FLAT
            if _average < average or _average < self.min_price:
                return GroupType.LOW
            elif _average > 1.5 * average:
                return GroupType.HIGH
            else:
                return GroupType.MID

        for i in range(1, len(self.prices)):
            if i == today_len:
                std_dev = std_dev_tomorrow
                average = average_tomorrow
            if abs(self.prices[i] - self.prices[current_group[-1]]) <= std_dev and self.prices[i] not in self.non_hours:
                current_group.append(i)
            else:
                group_type = __set_group_type(mean([self.prices[j] for j in current_group]), len(current_group) == 24,
                                              average)
                continuous_groups.append(Group(group_type, current_group))
                current_group = [i]
        group_type = __set_group_type(mean([self.prices[j] for j in current_group]), len(current_group) == 24, average)
        continuous_groups.append(Group(group_type, current_group))
        self.groups = continuous_groups

    def _find_group(self, index: int) -> Group:
        for group in self.groups:
            if index in group.hours:
                return group
        return Group(GroupType.UNKNOWN, [])


# -----------------------------
# prices = [0.18, 0.17, 0.16, 0.15, 0.15, 0.16, 0.18, 0.2, 0.21, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.23, 0.27,
#           0.29, 0.27, 0.25, 0.23, 0.22, 0.21]
# prices_tomorrow = [0.23, 0.23, 0.23, 0.23, 0.24, 0.25, 0.28, 0.29, 0.33, 0.34, 0.51, 0.77, 0.72, 0.49, 0.53, 0.75, 0.82,
#                    0.88, 0.93, 0.91, 0.87, 0.64, 0.44, 0.36]
#
# nb = NextWaterBoost()
# _dt: datetime = datetime.now().replace(hour=13, minute=32)
#
# tt2 = nb.next_predicted_demand(prices, prices_tomorrow, min_price=0.1, temp=40.2, temp_trend=0, target_temp=40,
#                                now_dt=_dt, demand=DEMAND_MINUTES[HvacPresets.Normal][Demand.NoDemand],
#                                non_hours=[7, 11, 12, 15, 16, 17, 22, 23])
# print("result", tt2)
# testlist = []


# for i in range(0,20):
#         dt = _dt + timedelta(minutes=i)
#         tt2 = nb.next_predicted_demand(prices, prices_tomorrow, min_price=0.1, temp=22.1, temp_trend=0, target_temp=40, now_dt=dt, demand=0, non_hours=[7, 11, 12, 15, 16, 17,22,23])
#         testlist.append((dt.minute, tt2))

# for t in testlist:
#     print(t[0], t[1])

