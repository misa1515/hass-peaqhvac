import logging
import homeassistant.helpers.template as template

_LOGGER = logging.getLogger(__name__)

NORDPOOL = "nordpool"

class NordPoolUpdater:
    def __init__(self, hass, hub):
        self._hass = hass
        self._hub = hub
        self.currency: str = ""
        self._prices: list = []
        self._prices_tomorrow: list = []
        self._state: float = 0
        self.nordpool_entity: str = ""
        self._setup_nordpool()

    @property
    def state(self) -> float:
        return self._state

    @state.setter
    def state(self, val) -> None:
        if self._state != val:
            self._hub.observer.broadcast("prices changed")
        self._state = val

    @property
    def prices(self) -> list:
        return self._prices

    @property
    def prices_tomorrow(self) -> list:
        return self._prices_tomorrow

    def update_nordpool(self):
        ret = self._hass.states.get(self.nordpool_entity)
        if ret is not None:
            try:
                ret_attr = list(ret.attributes.get("today"))
                if 23 <= len(ret_attr) <= 25:
                    self._prices = ret_attr
                else:
                    _LOGGER.error(f"Nordpool returned a faulty length of prices for today ({len(ret_attr)})")
            except Exception as e:
                _LOGGER.exception(f"Could not parse today's prices from Nordpool. Unsolveable error. {e}")
                return
            try:
                ret_attr_tomorrow = list(ret.attributes.get("tomorrow"))
                if (23 <= len(ret_attr_tomorrow) <= 25) or len(ret_attr_tomorrow) == 0:
                    self._prices_tomorrow = ret_attr_tomorrow
                else:
                    _LOGGER.error(f"Nordpool returned a faulty length of prices for tomorrow ({len(ret_attr_tomorrow)})")
            except Exception as e:
                _LOGGER.warning(f"Couldn't parse tomorrow's prices from Nordpool. Array will be empty. {e}")
                self._prices_tomorrow = []

            ret_attr_currency = str(ret.attributes.get("currency"))
            self.currency = ret_attr_currency
            self.state = ret.state
        else:
            _LOGGER.error("could not get nordpool-prices")

    def _setup_nordpool(self):
        try:
            entities = template.integration_entities(self._hass, NORDPOOL)
            if len(list(entities)) < 1:
                raise Exception("no entities found for Nordpool.")
            if len(list(entities)) == 1:
                self.nordpool_entity = list(entities)[0]
                _LOGGER.debug(f"Nordpool has been set up and is ready to be used with {self.nordpool_entity}")
                self.update_nordpool()
            else:
                raise Exception("more than one Nordpool entity found. Cannot continue.")
        except Exception as e:
            msg = f"Peaqhvac was unable to get a Nordpool-entity. Disabling Priceawareness: {e}"
            _LOGGER.error(msg)
