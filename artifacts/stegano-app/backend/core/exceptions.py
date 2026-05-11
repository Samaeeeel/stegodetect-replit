"""
Excepciones personalizadas del sistema.
Mantener las excepciones separadas permite manejar errores de forma uniforme
en toda la aplicación sin acoplar la lógica de negocio al framework HTTP.
"""


class ImageValidationError(Exception):
    """Se lanza cuando la imagen subida no supera las validaciones."""
    pass


class ModelInferenceError(Exception):
    """Se lanza cuando ocurre un error durante la inferencia del modelo."""
    pass


class ReportGenerationError(Exception):
    """Se lanza cuando no se puede generar el reporte PDF."""
    pass


class AnalysisNotFoundError(Exception):
    """Se lanza cuando se solicita un análisis que no existe en el registro."""
    pass
