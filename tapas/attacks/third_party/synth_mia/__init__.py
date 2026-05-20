"""Synth-MIA: A library for membership inference attacks on synthetic data."""

import pkgutil
import importlib
import inspect
from .base import BaseAttacker

__version__ = "0.1.0"

# Core imports
from .base import BaseAttacker
from .evaluation import AttackEvaluator
from .utils import (
    create_random_equal_dfs,
    fit_transformer,
)

# Initialize exports list with core components
__all__ = [
    "BaseAttacker",
    "AttackEvaluator", 
    "create_random_equal_dfs",
    "fit_transformer",
]

# Dynamic discovery of attacker classes
def _discover_attackers(package_path, package_name):
    """Dynamically discover all attacker classes that inherit from BaseAttacker."""
    attackers = {}
    
    for loader, module_name, is_pkg in pkgutil.iter_modules(package_path):
        if module_name.startswith('_'):  # Skip private modules
            continue
            
        try:
            # Import the module
            module = importlib.import_module(f"{package_name}.{module_name}")
            
            # Find all classes that inherit from BaseAttacker
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (obj != BaseAttacker and 
                    issubclass(obj, BaseAttacker) and 
                    obj.__module__ == module.__name__):
                    attackers[name] = obj
                    
        except ImportError as e:
            # Skip modules that can't be imported (missing dependencies, etc.)
            continue
    
    return attackers

# Discover attackers from the attackers subpackage
try:
    from . import attackers
    discovered_attackers = _discover_attackers(attackers.__path__, f"{__name__}.attackers")
    
    # Add discovered attackers to globals and __all__
    for name, attacker_class in discovered_attackers.items():
        globals()[name] = attacker_class
        __all__.append(name)
        
except ImportError:
    # Handle case where attackers subpackage doesn't exist
    pass
    
# Also import other modules dynamically (non-attacker modules)
for loader, module_name, is_pkg in pkgutil.iter_modules(__path__):
    if module_name not in ['base', 'evaluation', 'utils', 'attackers']:
        try:
            module = importlib.import_module(f"{__name__}.{module_name}")
            globals()[module_name] = module
            __all__.append(module_name)
        except ImportError:
            continue
