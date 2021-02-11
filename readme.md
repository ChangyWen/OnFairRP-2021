# OnFairRP: Online Fair Ride-Sharing Pricing
<p align="center">
    <img src="readme_fig/fig1-cropped.png" border="0" width=50%/>
</p>
<b><p align="center">
    Figure 1. Process of ride-sharing pricing
</p></b>
<br>
<p align="center">
    <img src="readme_fig/fig2-cropped.png" border="0" width=50%/>
</p>
<b><p align="center">
    Figure 2. Motivating example of pricing without fairness
</p></b>

## Overview
* Ride-sharing is a transportation mode in which individual travelers share a vehicle for a trip. <br>
* OnFairRP is an online fair ride-sharing pricing algorithm <br>
* This project includes a ride-sharing simulator, the implementation of OnFair and other pricing schemes for comparison. <br>

## Fairness
The motivating example in Figure 2 captures the real fairness issue in ride-sharing pricing. Though Alice and Bob have similar origins and same destination, Alice will be charged less since Alice’s neighborhood has more potential passengers to share the ride, which is unfair to Bob. 

To deal with this issue, we consider the notion of **individual fairness**. Specifically, if two riders are similar (e.g., in terms of origin, destination, and departure time), they should receive similar discounts. Our algorithm, OnFairRP, guarantees individual fairness among riders in an online way. That is, for two riders arriving at different times, the difference between the discount of the latter rider and the one of any previous rider is bounded by *K* times the similarity of the two riders.

## Dependencies
* scipy
* networkx
* [osmnx](https://osmnx.readthedocs.io/en/stable/): e.g., to get a networkx graph of Chengdu, China:
    ```python
    import osmnx as ox
    G = ox.graph_from_place('Chengdu, China', network_type='drive')
    ox.plot_graph(G, node_size=0)
    ```
    <p align="center">
        <img src="readme_fig/Chengdu.png" border="0" width=45%/>
    </p>
* To install all the dependent packages:
    ```bash
    pip install -r requirements.txt
    ```

## Data Source
* Real-world data traces are obtained from [DiDi Chuxing GAIA Initiative](https://gaia.didichuxing.com).

## Usage (run with arguments specified)
* To price (i.e., decide discounts) for requests using OnFairRP algorithm:
  ```bash
  python main_pricing.py --scheme OnFair --a 0.99 --K 0.32 --omega 0.4 --week 1 --isLinear
  ```
* To simulate the ride-sharing process with the ride requests and their corresponding price:
  ```bash
  python main.py --scheme OnFair --a 0.99 --K 0.32 --omega 0.4 --week 2 --isLinear
  ```
* To learn more details of arguments of `main.py` or `main_pricing.py`:
  ```bash
  python main.py --help
  ```


