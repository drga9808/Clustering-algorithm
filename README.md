# Clustering-algorithm

This clustering algorithm I am proposing is based on the <b>Linde, Buzo, and Gray (LBG)</b> algorithm -- Generalized Llyod algorithm, the original paper can be found in [DOI: 10.1109/TCOM.1980.1094577](https://ieeexplore.ieee.org/document/1094577). I am trying to make an automatic clustering algorithm that can find an optimal number and positions of clusters based on the distribution of the data points. 

I based my work on the book "Fundamentals of Internet of Things" by Sudhir Kumar in [DOI: 10.1201/9781003225584](https://www.taylorfrancis.com/books/mono/10.1201/9781003225584/fundamentals-internet-things-sudhir-kumar). In chapter 3 he proposes a clustering technique, I made some modifications. I created the Python code for computation and visualization in 2 Dimensions. 

<b>Problems:</b>
- Fix the situation when an added cluster has no points assigned to it.
- Prove with more than 100 points.
- Try to show the boundaries without overlapping one with the others.
- Verify the stop criterium.
