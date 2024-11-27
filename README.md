# Mathematical Model of Love 

## Project Description

This project, based on Strogatz's mathematical model of love, analyzes if people with the same romantic style attract each other. It simulates and visualizes the dynamics of Romeo and Juliet’s relationship using  mathematical models with systems of differential equations. The models investigate how individuals with similar romantic tendencies (as defined by their parameters) might behave in a relationship over time using eigenvectors and phase portraits.

### Models:

1. **Model I**: 
   -  R' = -3R - 2J
   -  J' = -2R - 3J
   
   - This model represents the dynamics of a relationship where both Romeo (R) and Juliet (J) exhibit **mutual indifference**. Over time, their feelings die out, and all that remains is mutual indifference.
   - Both Romeo and Juliet have "Hermit" romantic style; Romeo retreats from his own feelings as well as Juliet’s. Juliet retreats from her own feelings as well as Romeo's.

3. **Model II**:
   - R' = -R + 4J 
   - J' = 4R - J
   
   - This model explores a scenario where Romeo and Juliet either **fall deeply in love** or **end up despising each other**, depending on their initial conditions and the fact that they share the same romantic style.
   - Both Romeo and Juliet have "Cautious Lover" romantic style; Romeo retreats from his own feelings but is encouraged by Juliet’s. Juliet retreats from her own feelings but is encouraged by Romeo’s.

## Features

- **Differential Equation Solver**: The project uses the `scipy` library to solve systems of differential equations.
- **Eigenvalue and Eigenvector Calculation**: Eigenvalues and eigenvectors are calculated for each model to interpret the stability of the system.
- **Phase Portraits**: The phase portrait is generated for both models, illustrating the relationship dynamics based on different initial conditions.
- **Time Evolution**: The script simulates and plots the evolution of Romeo and Juliet’s relationship over time, based on their shared romantic styles.

## Prerequisites

To run this project, you need to have Python 3 and the following libraries installed:

- `numpy` – For numerical operations
- `scipy` – For solving differential equations
- `matplotlib` – For plotting graphs

## Solution

Graphs and phase portrait outputs from the python code are included in the folder to display the results.


