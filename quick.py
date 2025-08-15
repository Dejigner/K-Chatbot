# import fitz
# print(fitz.__doc__)
# print(fitz.__version__)

import fitz  # PyMuPDF
print("fitz path:", getattr(fitz, "__file__", "<none>"))
print("has open:", hasattr(fitz, "open"))
print("version:", getattr(fitz, "__version__", "<none>"))
print("doc head:", (getattr(fitz, "__doc__", "") or "")[:120])