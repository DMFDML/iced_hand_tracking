# ICED Hand Tracking Demo

This repo contains the source code for the ICED non-intrusive low-cost easy-configurable hand tracking demo.

**N.b.** The demo expects three cameras (full software will allow n cameras).


## Setup

Creating the Virtual Environment
```
python -m venv .venv
```

Load the Virtual Environment
```
.venv\Scripts\activate
```

Install the packages for the code to run.
```
pip install -r requirements.txt
```

## Running the tool.

Step 1. Calibrate the scene.

Make sure the cameras and VR headset are connected. Place the aruco marker on the headset to act as the datum point for the positioning data. From the top-level folder run defining the camera ids as a comma seperated list.

```
python calibrate\main.py 1,2,3
```

N.b. You may have to change the camera ids if one of the cameras is not selected.

The script should run for 5 mins (this can be changed in calibrate/main.py). Showing the camera feeds. Move the headset with around the space. The 3d plot should update showing the positions you have captured. The script will timeout resulting in a `input_tensors.csv` and `output_tensors.csv` in the `tmp` directory.

Step 2. Train the ML.

Run the command in the top-level directory.
```
python train\main.py
```

This will train a set of ML models, which are saved to the `tmp` directory. The RMSE score will be printed on the screen.

Step 3. Run the tracking module

From the top-level folder run the following command defining the camera ids as a comma seperated list.

```
python track\main.py 1,2,3
```