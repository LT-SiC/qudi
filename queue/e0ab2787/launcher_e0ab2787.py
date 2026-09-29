
import importlib.util, traceback

TARGET = r"C:\src\qudi\queue\e0ab2787\SSR_DDRF_protected_readout.py"
try:
    spec = importlib.util.spec_from_file_location("embedded_gui_task", TARGET)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if hasattr(mod, "run_fun") and callable(mod.run_fun):
        mod.run_fun()
    else:
        for name in dir(mod):
            obj = getattr(mod, name)
            if isinstance(obj, type) and hasattr(obj, "run_fun"):
                obj().run_fun()
                break
        else:
            print("[Launcher] No run_fun found.")
except Exception:
    traceback.print_exc()
