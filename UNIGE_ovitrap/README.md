## ACCESS GOOGLE EARTH ENGINE

1. Subscribe to GEE: https://signup.earthengine.google.com
2. Explore available datasets: https://developers.google.com/earth-engine/datasets/
3. Read the API docs: https://developers.google.com/earth-engine

## VISUALIZE AND DOWNLOAD DATA

There are 3 options:

#### 1. GEE Code Editor
Use the code editor: https://code.earthengine.google.com/ 

Commands are in JavaScript, see documentation here: https://developers.google.com/earth-engine/playground

#### 2. Python API
I find the GEE code editor quite cumbersome, so I wrote some code in Python that allows for more flexibility.
If you want to use it as well, follow these steps: 

4. Install Python 3.7.4, if you don't have it already: https://www.python.org/downloads/
5. Install the GEE Python API (and required auxiliary software): https://developers.google.com/earth-engine/python_install
6. Install geetools: https://pypi.org/project/geetools/
```python
pip install geetools
```
7. Execute the python script 'get_data.py'. In Linux/MacOS:
```python
python get_data.py
```
That's it! you should see images appearing in the directory where 'get_data.py' is running.
To get all needed data, you need to change the script accordingly; see the comment lines in the code to understand what needs to be changed.

#### 3. QGIS plugin
There's also a QGIS plugin for GEE. I haven't tested it and it seems to be still under development, but it might be worth to try:
https://github.com/gee-community/qgis-earthengine-plugin
Thanks Fleur Hierink for pointing that out!

#### N.B. different datasets have different date ranges, spatial/temporal resolution and variable names, check carefully the datasets descriptions at https://developers.google.com/earth-engine/datasets/

in case of doubts, drop me (Jacopo Margutti) a message at jmargutti@redcross.nl
