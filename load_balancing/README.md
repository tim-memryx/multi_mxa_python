# One Application using Different DFPs on Separate M.2s

Here's an example of running MiDAS Depth + Cartoonizer on separate M.2s, but within a single application that uses multiple AsyncAccl objects.


## Requirements

Make sure you've activated your memryx python venv and have installed the `opencv-python` pip package.


## Run

```python
python3 run.py -d 0 -c 2
```

This will use MXA 0 for Depth and MXA 1 for Cartoonizer. Edit run.py if you want to try different M.2s in your system.

The `-d` argument is the camera index for Depth and `-c` is the camera index for Cartoonizer.



## How Do They Work?

Same as before, to enable the apps to run in parallel, we are simply setting the `group_id` argument for AsyncAccl to the index of the MXA M.2 module we want to use.

In this case, we're using two AsyncAccl objects within the same Python application, instead of two completely separate applications.

