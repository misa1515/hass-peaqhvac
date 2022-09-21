from custom_components.peaqhvac.service.models.hvacbrands import HvacBrand
from custom_components.peaqhvac.service.models.config_model import ConfigModel
from custom_components.peaqhvac.service.hvac.ihvac import IHvac
from custom_components.peaqhvac.service.hvac.hvactypes.ivt import IVT
from custom_components.peaqhvac.service.hvac.hvactypes.nibe import Nibe
from custom_components.peaqhvac.service.hvac.hvactypes.thermia import Thermia
from homeassistant.core import (
    HomeAssistant
)


class HvacFactory:
    HVACTYPES = {
        HvacBrand.Nibe: Nibe,
        HvacBrand.IVT: IVT,
        HvacBrand.Thermia: Thermia
    }

    @staticmethod
    def create(hass: HomeAssistant, options: ConfigModel, hub) -> IHvac:
        return HvacFactory.HVACTYPES[options.hvacbrand](hass, hub)