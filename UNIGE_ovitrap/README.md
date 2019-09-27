# ACCESS GOOGLE EARTH ENGINE DATA

How to access Google Earth Engine (GEE) data?

1. Subscribe to GEE: https://signup.earthengine.google.com
2. Explore available datasets: https://developers.google.com/earth-engine/datasets/
3. Read the API docs: https://developers.google.com/earth-engine

data can be visualized and downloaded to Google Drive using the code editor: https://code.earthengine.google.com/

I find it quite cumbersome, so I wrote some code in Python that allows for more flexibility.
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

that's it! you should see images appearing in the directory where 'get_data.py' is running.
to get all needed data, you need to change the script accordingly; see the comment lines in the code to understand what needs to be changed.
#### N.B. different datasets have different date ranges, spatial/temporal resolution and variable names, check carefully the datasets descriptions at https://developers.google.com/earth-engine/datasets/

in case of doubts, drop me (Jacopo Margutti) a message at jmargutti@redcross.nl
