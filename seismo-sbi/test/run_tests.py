import os

omitPaths=["*__init__.py","*test_*.py","*_virtualenv.py","/usr/*","*/venv/*"]

OMIT = ",".join(omitPaths)

os.system(f"coverage run -m -a --omit={OMIT} unittest discover input_settings/")
os.system(f"coverage run -m -a --omit={OMIT} unittest discover outputs/")
os.system(f"coverage run -m -a --omit={OMIT} unittest discover wrapper/")

os.system("coverage html")
os.system("coverage report")

os.system("coverage erase")
