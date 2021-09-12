# Video Face Cropper
<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#result">Result</a></li>

  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project


<!-- GETTING STARTED -->
## Getting Started

Just copy and paste this codes to install dependencies and run the project

### Prerequisites

create an environment with `python 3.8`
  ```sh
  conda create -n "ENVIRONMENT_NAME" python=3.8
  ```

### Installation

1. Clone the repo
   ```sh
   git clone 
   ```
      
2. Install Python libraries
   ```sh
   pip install opencv-python
   pip install ffmpeg-python
   pip install batch_face
   ```

<!-- USAGE EXAMPLES -->
## Usage
1. Import your videos in `videos` directory
2. Run create_frame.py to create frames data. You can see created frames in `frames` directory.
```
python create_frames,py
```
3. Run face_detection.py to detect faces in created frames and crop them.
```
python face_detection,py
```


## Result

