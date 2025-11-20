from core.meta import InterfaceMetaclass


class FilteredCounterInterface(metaclass=InterfaceMetaclass):
    """
    Interface für das FilteredCounterMeasurement-Modul.
    Legt fest, welche Methoden in der Logic vorhanden sein müssen.
    """

    def refilter(self, threshold=None, n_filter=None):
        """Anwendung neuer Filterparameter auf vorhandene Rohdaten."""
        pass

    def get_result(self):
        """Gibt das aktuelle Ergebnis als dict zurück."""
        pass

    def save(self, filepath):
        """Speichert das Histogramm unter dem gegebenen Pfad."""
        pass

    
    def __init__(self,  **kwargs):
        pass

    def on_activate(self):
        pass

    def on_deactivate(self):
        pass
    
    def _init_plot(self):
        pass

    def prepare(self):
        pass

    def run(self):
        pass

    def _apply_filter_and_histogram(self):
        pass

    def _update_plot(self):
        if self.histogram is not None and self.bin_edges is not None:
            self.plot_curve.setData(x=self.bin_edges[:-1], y=self.histogram)

    def refilter(self, threshold=None, n_filter=None):
        pass

    def get_result(self):
        pass

    def save(self, filepath):
        pass

    def cleanup(self):
        pass

        