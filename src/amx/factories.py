"""Factory classes for model creation."""

from typing import Any, Dict, Optional, Type

from amx.core.exceptions import ConfigurationError
from amx.core.interfaces import FluidModel, MixingModel, RheologyModel, TurbulenceModel
from amx.physics.models import (
    ComprehensiveMixingModel,
    FluidProperties,
    FluidType,
    NewtonianFluid,
    NonNewtonianFluid,
    RheologyModelType,
    StandardKEpsilon,
    TurbulenceModelType,
)


class FluidFactory:
    """Factory for creating fluid models."""
    
    @staticmethod
    def create(fluid_type: FluidType, **kwargs) -> FluidModel:
        """
        Create fluid model.
        
        Args:
            fluid_type: Type of fluid
            **kwargs: Model parameters
            
        Returns:
            Fluid model instance
        """
        if fluid_type == FluidType.WATER:
            props = FluidProperties(
                temperature=kwargs.get('temperature', 298.15),
                density=kwargs.get('density', 998.0),
                dynamic_viscosity=kwargs.get('viscosity', 0.001),
            )
            return NewtonianFluid(props)
        
        elif fluid_type == FluidType.SLUDGE:
            return NonNewtonianFluid(
                total_solids=kwargs.get('total_solids', 3.5),
                temperature=kwargs.get('temperature', 308.15),
                model_type=kwargs.get('rheology_model', RheologyModelType.HERSCHEL_BULKLEY),
            )
        
        elif fluid_type == FluidType.CUSTOM:
            if 'properties' in kwargs:
                return NewtonianFluid(kwargs['properties'])
            else:
                raise ConfigurationError("Custom fluid requires 'properties' parameter")
        
        else:
            raise ConfigurationError(f"Unknown fluid type: {fluid_type}")
    
    @staticmethod
    def from_config(config: Dict[str, Any]) -> FluidModel:
        """
        Create fluid model from configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Fluid model instance
        """
        fluid_type = FluidType(config.get('type', 'water'))
        return FluidFactory.create(fluid_type, **config)


class TurbulenceFactory:
    """Factory for creating turbulence models."""
    
    _models: Dict[TurbulenceModelType, Type[TurbulenceModel]] = {
        TurbulenceModelType.K_EPSILON: StandardKEpsilon,
        # Add more models as implemented
    }
    
    @staticmethod
    def create(model_type: TurbulenceModelType, 
              kinematic_viscosity: float, **kwargs) -> TurbulenceModel:
        """
        Create turbulence model.
        
        Args:
            model_type: Type of turbulence model
            kinematic_viscosity: Kinematic viscosity [m²/s]
            **kwargs: Additional model parameters
            
        Returns:
            Turbulence model instance
        """
        if model_type not in TurbulenceFactory._models:
            raise ConfigurationError(
                f"Turbulence model {model_type} not implemented. "
                f"Available: {list(TurbulenceFactory._models.keys())}"
            )
        
        model_class = TurbulenceFactory._models[model_type]
        
        if model_type == TurbulenceModelType.K_EPSILON:
            return model_class(kinematic_viscosity)
        else:
            # For future models with different constructors
            return model_class(kinematic_viscosity, **kwargs)
    
    @staticmethod
    def from_config(config: Dict[str, Any], 
                   kinematic_viscosity: float) -> TurbulenceModel:
        """
        Create turbulence model from configuration.
        
        Args:
            config: Configuration dictionary
            kinematic_viscosity: Kinematic viscosity [m²/s]
            
        Returns:
            Turbulence model instance
        """
        model_type = TurbulenceModelType(config.get('model', 'k-epsilon'))
        return TurbulenceFactory.create(model_type, kinematic_viscosity, **config)


class MixingFactory:
    """Factory for creating mixing models."""
    
    @staticmethod
    def create(fluid_model: FluidModel) -> MixingModel:
        """
        Create mixing model.
        
        Args:
            fluid_model: Fluid model to use
            
        Returns:
            Mixing model instance
        """
        return ComprehensiveMixingModel(fluid_model)
    
    @staticmethod
    def from_config(config: Dict[str, Any], 
                   fluid_model: FluidModel) -> MixingModel:
        """
        Create mixing model from configuration.
        
        Args:
            config: Configuration dictionary
            fluid_model: Fluid model to use
            
        Returns:
            Mixing model instance
        """
        # For now, we only have one mixing model
        # Can be extended with different models in the future
        return MixingFactory.create(fluid_model)


class ModelFactory:
    """Main factory for creating all models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model factory.
        
        Args:
            config: Complete configuration dictionary
        """
        self.config = config
        self._fluid_model = None
        self._turbulence_model = None
        self._mixing_model = None
    
    @property
    def fluid(self) -> FluidModel:
        """Get or create fluid model."""
        if self._fluid_model is None:
            fluid_config = self.config.get('fluid', {})
            self._fluid_model = FluidFactory.from_config(fluid_config)
        return self._fluid_model
    
    @property
    def turbulence(self) -> TurbulenceModel:
        """Get or create turbulence model."""
        if self._turbulence_model is None:
            turb_config = self.config.get('turbulence', {})
            
            # Get kinematic viscosity from fluid model
            if hasattr(self.fluid, 'props'):
                nu = self.fluid.props.kinematic_viscosity
            else:
                # For non-Newtonian, use a representative value
                nu = 1e-6
            
            self._turbulence_model = TurbulenceFactory.from_config(turb_config, nu)
        return self._turbulence_model
    
    @property
    def mixing(self) -> MixingModel:
        """Get or create mixing model."""
        if self._mixing_model is None:
            mixing_config = self.config.get('mixing', {})
            self._mixing_model = MixingFactory.from_config(mixing_config, self.fluid)
        return self._mixing_model
    
    def create_all(self) -> Dict[str, Any]:
        """
        Create all models.
        
        Returns:
            Dictionary of all models
        """
        return {
            'fluid': self.fluid,
            'turbulence': self.turbulence,
            'mixing': self.mixing,
        }