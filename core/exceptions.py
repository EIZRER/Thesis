class PipelineError(Exception):
    """General pipeline error."""
    pass


class COLMAPError(PipelineError):
    """COLMAP step failed."""
    pass


class OpenMVSError(PipelineError):
    """OpenMVS step failed."""
    pass


class ConfigError(PipelineError):
    """Configuration or executable not found."""
    pass
