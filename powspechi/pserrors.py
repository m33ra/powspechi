class Error(Exception):
    """Base class for other exceptions."""
    # Based on https://www.programiz.com/python-programming/user-defined-exception
    pass

class SupmapError(Error):
    """When one is dealing with a single map."""
    pass

class IsomapError(Error):
    """When the iso file for edge correction does not exist."""
    pass

class PowSpecError(Error):
    """When the averaged normalized spectrum does not exist."""
    pass