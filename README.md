# DRLHV-1: Customizing the OpenAI Gym Taxi Environment

## Overview

I changed some **parameters** on the **gym** library (**especially** the `taxi.py` file) which was published by **OpenAI**. **Differences** emerge around map size, station count, new **obstacles**, etc. For real learning and having a deep **intuitive** approach, I changed some of these values and also some of the **affected** functions.

## Main Changes

- **Map size:** 5x5 to 6x6
- **Station Count:** 4 to 5
- **Added new obstacle "X"** as an **unmovable** part of the grid

## Challenges & Observations

The **hardest** part was the **Render** part because a lot of **things** were **affected** when the map size changed.

These changes **required** some **function** changes, mostly due to static coding, which also **surprised** me:

> ...I **didn't** expect such a basic coding approach (e.g., static values like '5' **representing** row size when we also have a row-size variable???) from OpenAI.

## Data Persistence

I need to keep the **training** data as both a **CSV** file for humans and an **NPY** file for machines. So I coded basic Python **structures** to keep the files and, if **needed**, read the file.

## Setup
```bash
git clone https://github.com/MuhammetAliKaya/DRLHV-1.git
cd DRLHV-1
python -m venv .venv
.\.venv\Scripts\Activate
pip install "numpy<2" gym pygame
python RL.py


press [e]     or     [h]
press [enter]
```

## Resault

<p align="center">
  <img src="resault.gif" width="500" />
  <br>
</p>
