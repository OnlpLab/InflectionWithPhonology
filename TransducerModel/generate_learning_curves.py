import pandas as pd
from matplotlib import pyplot as plt

pairs = ['Bulgarian-Adj', 'Bulgarian-V', 'Finnish-Adj', 'Finnish-N', 'Finnish-V',
         'Hungarian-V', 'Georgian-N', 'Georgian-V', 'Latvian-N', 'Latvian-V',
         'Albanian-V', 'Swahili-Adj', 'Swahili-V', 'Turkish-Adj', 'Turkish-V']

# Read tables values
filename = 'results.xlsx'
xl_file = pd.ExcelFile(filename)

tables = []
for sheet_name in pairs:
    # Read the sheet data into a pandas DataFrame
    sheet_data = xl_file.parse(sheet_name)

    # Extract the table data by selecting the first 9 rows and the first 5 columns
    table = sheet_data.iloc[:9, :4].round(3)  # .values.tolist()
    tables.append(table)

# Construct graphs
column_names = ['Baseline', 'Data Manip.', 'Model Manip.']
x_values = range(1000, 8001, 1000)

fig, axs = plt.subplots(5, 3, figsize=(9, 12))

fig.text(0.5, 0.043, '# Train Samples', ha='center', fontsize=14)
fig.text(0.01, 0.5, 'Test Accuracy', va='center', rotation='vertical', fontsize=14)
fig.subplots_adjust(left=0.5, bottom=0.5)

axs = axs.flatten()

for i, table in enumerate(tables):
    ax = axs[i]

    ax.set_ylim([0, 1.0])

    # Plot the values for each column: lines and dots
    for column_name in column_names:
        ax.plot(x_values, table[column_name], label=column_name)
        ax.scatter(x_values, table[column_name], marker='o', s=10)

    # Add labels only to the left-most and bottom-most subplots
    if i // 3 != 4:
        ax.set_xticklabels([])

    if i % 3 != 0:
        ax.set_yticklabels([])

    ax.set_title(pairs[i])

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=len(column_names), bbox_to_anchor=(0.5, 0.005))

# Adjust the subplot layout
plt.tight_layout()

plt.show()

# Note: the resulted figure requires further adjustments. Use the GUI 'configure subplots' option
# to modify the left & bottom borders:
# top=0.971,
# bottom=0.087,
# left=0.077,
# right=0.977,
# hspace=0.231,
# wspace=0.1