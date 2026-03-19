![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-orange)

# Multi-Robot Collision Avoidance with Probabilistic Mahalanobis Distance Constraints
We present a probabilistic Mahalanobis distance constraint combined with MPPI algorithm to realize multi-robot system collision avoidance.

<video src="https://github.com/CreedonChan/MRCA-PMDC/blob/main/experiment_video.mp4" width="720" controls>
  Your browser does not support the video tag.
</video>

## Environment Setup and Installation
### System Requirements

OS: Recommended [Ubuntu 20.04 / Windows 10 / macOS]
Python: [3.8 / 3.9 / 3.10]
Hardware: NVIDIA GPU + CUDA [11.8/12.1] recommended (CPU only also supported)

| Package | Version | Purpose |
|:---|:---|:---|
| torch | 2.7.1 | Deep learning framework |
| numpy | 2.3.1 | Numerical computing |
| pandas | 2.3.1 | Data processing |
| matplotlib | 3.10.3 | Visualization |
| pillow | 11.3.0 | Image processing |
| scipy | 1.16.0 | Scientific computing |

### ⚙️ Installation Steps

```bash
# 1. Clone the repository
git clone https://github.com/CreedonChan/MRCA-PMDC.git
cd MRCA-PMDC

# 2. Create and activate virtual environment
# For Linux/macOS:
python3 -m venv venv
source venv/bin/activate
# For Windows:
# python -m venv venv
# venv\Scripts\activate

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## Simulation Demos
<table>
  <tr>
    <td align="center">
      <strong>Symmetrical Scenario</strong><br>
      <strong>(10 robots)</strong>
    </td>
    <td align="center">
      <strong>Static Obstacle Scenario</strong><br>
      <strong>(8 robots, 11 obstacles)</strong>
    </td>
    <td align="center">
      <strong>Dynamic Obstacle Scenario</strong><br>
      <strong>(7 robots)</strong>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/CreedonChan/MRCA-PMDC/blob/main/10_robots_and_0_obs.gif?raw=true" width="300">
    </td>
    <td align="center">
      <img src="https://github.com/CreedonChan/MRCA-PMDC/blob/main/8_robots_and_11_obs.gif?raw=true" width="300">
    </td>
    <td align="center">
      <img src="https://github.com/CreedonChan/MRCA-PMDC/blob/main/7_robots_and_0_obs.gif?raw=true" width="300">
    </td>
  </tr>
  <tr>
    <td align="center">
      <em>Symmetrical starting points<br>to diagonal targets</em>
    </td>
    <td align="center">
      <em>Navigation in cluttered<br>environment with static obstacles</em>
    </td>
    <td align="center">
      <em>Robots as dynamic obstacles<br>for each other</em>
    </td>
  </tr>
</table>

## Experiment Video
https://github.com/CreedonChan/MRCA-PMDC/blob/main/experiment_video.mp4


