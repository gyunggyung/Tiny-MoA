
import warnings
# Suppress ResourceWarning universally for this package
warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", message=r"unclosed file <_io.TextIOWrapper name='nul'", category=ResourceWarning)
