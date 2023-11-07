import numpy as np
import matplotlib.pyplot as plt

def plot_data(X, CH=None, Show_Point_labels=False, show_CHs=False, show_line=False, 
              boundaries_points=None, boundaries_lines_m=None, boundaries_lines_b=None,
              title = None):
    # Scatter plot for X points
    plt.scatter(X[:,0], X[:,1], color='blue', marker='o')
    
    # Annotate each point in X with its coordinates if requested
    if Show_Point_labels:
        for x, y in X:
            plt.annotate(f'({x}, {y})', (x, y), textcoords="offset points", xytext=(0,5), ha='center')
            
    # Scatter plot for CH points if provided
    if CH is not None and show_CHs:
        plt.scatter(CH[:,0], CH[:,1], color='red', marker='o')
        
        # Annotate the CH points
        for x, y in CH:
            plt.annotate(f'({x}, {y})', (x, y), textcoords="offset points", xytext=(0,5), ha='center')
    
    # Plotting the boundary line if requested
    if boundaries_points is not None and show_line:
        x_vals = np.linspace(min(X[:, 0]) - 1, max(X[:, 0]) + 1, 400)
        
        # Loop through each boundary
        for index, (m, b) in enumerate(zip(boundaries_lines_m, boundaries_lines_b)):
            if m == float('inf'):
                # We assume that x_vals is a numpy array with the x-values where we want to plot the vertical line.
                # We need to draw a vertical line at the x-coordinate of the boundary point.
                x_val = boundaries_points[index][0]  # Use 'index' to access the correct boundary point
                y_vals = np.array([min(X[:, 1]), max(X[:, 1])])  # For vertical lines, y can take any value
                plt.axvline(x=x_val, color='green')  # This is how you'd typically plot a vertical line in matplotlib
            else:
                y_vals = m * x_vals + b
                plt.plot(x_vals, y_vals, 'green')  # Plot the non-vertical lines

                
    # Setting the x and y limits
    plt.xlim(min(X[:, 0]-2), max(X[:, 0]) + 1)
    plt.ylim(min(X[:, 1]-2), max(X[:, 1]) + 1)
    
    # Setting labels and title
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    if title == None:
        plt.title('Customizable Scatter Plot')
    else:
        plt.title(title)        
    # Display the plot
    plt.grid(True)
    plt.show()
