from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Tuple

from peaqevcore.common.models.observer_types import ObserverTypes

from custom_components.peaqhvac.service.hub.target_temp import adjusted_tolerances
from custom_components.peaqhvac.service.hvac.house_heater.house_heater_helpers import HouseHeaterHelpers
from custom_components.peaqhvac.service.hvac.house_heater.models.calculated_offset import CalculatedOffsetModel
from custom_components.peaqhvac.service.hvac.house_heater.models.offset_adjustments import OffsetAdjustments
from custom_components.peaqhvac.service.hvac.house_heater.temperature_helper import get_tempdiff_inverted, get_temp_extremas, get_temp_trend_offset
from custom_components.peaqhvac.service.hvac.interfaces.iheater import IHeater
from custom_components.peaqhvac.service.hvac.offset.offset_utils import adjust_to_threshold
from custom_components.peaqhvac.service.models.enums.demand import Demand

_LOGGER = logging.getLogger(__name__)

OFFSET_MIN_VALUE = -10


class HouseHeaterCoordinator(IHeater):
    def __init__(self, hvac):
        self._hvac = hvac
        self._degree_minutes = 0
        self._current_offset: int = 0
        self._offsets: dict = {}
        self._current_adjusted_offset: int = 0
        #self._temp_helper = HouseHeaterTemperatureHelper(hub=hvac.hub)
        self._helpers = HouseHeaterHelpers(hvac=hvac)
        super().__init__(hvac=hvac)

    @property
    def aux_offset_adjustments(self) -> dict:
        return self._helpers.aux_offset_adjustments

    @property
    def current_adjusted_offset(self) -> int:
        return int(self._current_adjusted_offset)

    @current_adjusted_offset.setter
    def current_adjusted_offset(self, val) -> None:
        if isinstance(val, (float, int)):
            self._current_adjusted_offset = val

    @property
    def is_initialized(self) -> bool:
        return True

    @IHeater.demand.setter
    def demand(self, val):
        self._demand = val

    @property
    def current_offset(self) -> int:
        return self._current_offset

    @current_offset.setter
    def current_offset(self, val) -> None:
        if isinstance(val, (float, int)):
            self._current_offset = val

    @property
    def current_tempdiff(self):
        return get_tempdiff_inverted(self.current_offset, self._hvac.hub.sensors.get_tempdiff(), self._current_tolerances)

    @property
    def turn_off_all_heat(self) -> bool:
        return self._hvac.hub.sensors.average_temp_outdoors.value > self._hvac.hub.options.heating_options.outdoor_temp_stop_heating

    def get_current_offset(self) -> Tuple[int, bool]:
        self._offsets = self._hvac.model.current_offset_dict_combined
        force_update: bool = False

        outdoor_temp = self._hvac.hub.sensors.average_temp_outdoors.value
        temp_diff = self._hvac.hub.sensors.get_tempdiff()
        
        if self.turn_off_all_heat or (self._hvac.hub.offset.max_price_lower(temp_diff)) and outdoor_temp >= 0:
            self._helpers.aux_offset_adjustments[OffsetAdjustments.PeakHour] = OFFSET_MIN_VALUE
            self.current_adjusted_offset = OFFSET_MIN_VALUE
            return OFFSET_MIN_VALUE, True
        else:
            self._helpers.aux_offset_adjustments[OffsetAdjustments.PeakHour] = 0

        offsetdata = self.get_calculated_offsetdata(force_update)
        force_update = self._helpers.temporarily_lower_offset(offsetdata, force_update)

        if self.current_adjusted_offset != round(offsetdata.sum_values(),0):
            ret = adjust_to_threshold(
                offsetdata,
                self._hvac.hub.sensors.average_temp_outdoors.value,
                self._hvac.hub.offset.model.tolerance
            )
            self.current_adjusted_offset = round(ret,0)
            if force_update:
                self._hvac.hub.observer.broadcast(ObserverTypes.UpdateOperation)
        return self.current_adjusted_offset, force_update

    def _get_demand(self) -> Demand:
        return self._helpers.helper_get_demand()

    def _current_tolerances(self, determinator: bool, current_offset: int, adjust_tolerances: bool = True) -> float:
        if adjust_tolerances:
            tolerances = adjusted_tolerances(
                current_offset,
                self._hvac.hub.sensors.set_temp_indoors.min_tolerance,
                self._hvac.hub.sensors.set_temp_indoors.max_tolerance
            )
        else:
            tolerances = self._hvac.hub.sensors.set_temp_indoors.min_tolerance, self._hvac.hub.sensors.set_temp_indoors.max_tolerance
        return tolerances[0] if (determinator > 0 or determinator is True) else tolerances[1]

    def get_calculated_offsetdata(self, force_update: bool = False) -> CalculatedOffsetModel:
        force_update = self._check_next_hour_offset(force_update=force_update)
        tempdiff = get_tempdiff_inverted(
                                         self.current_offset,
                                         self._hvac.hub.sensors.get_tempdiff(),
                                         self._current_tolerances
                                     )
        tempextremas = get_temp_extremas(
                                        self.current_offset,
                                        [self._hvac.hub.sensors.set_temp_indoors.adjusted_temp - t for t in self._hvac.hub.sensors.average_temp_indoors.all_values],
                                        self._current_tolerances
                                     )
        temptrend = get_temp_trend_offset(
                                         self._hvac.hub.sensors.temp_trend_indoors.is_clean,
                                         self._hvac.hub.predicted_temp,
                                         self._hvac.hub.sensors.set_temp_indoors.adjusted_temp
                                     )

        return CalculatedOffsetModel(current_offset=self.current_offset,
                                     current_tempdiff=tempdiff,
                                     current_temp_extremas=tempextremas,
                                     current_temp_trend_offset=temptrend)

    async def async_update_operation(self):
        pass

    def _check_next_hour_offset(self, force_update: bool) -> bool:
        if not len(self._offsets):
           return force_update
        hour = datetime.now().replace(minute=0, second=0, microsecond=0)
        if datetime.now().minute >= 55:
            hour += timedelta(hours=1)
        try:
            if self._hvac.hub.price_below_min(hour):
                _offset = max(self._offsets[hour],0)
            else:
                _offset = self._offsets[hour]
        except:
            _LOGGER.warning(
                "No Price-offsets have been calculated. Setting base-offset to 0."
            )
            _offset = 0
        if self.current_offset != _offset:
            force_update = True
            self.current_offset = _offset
        return force_update




