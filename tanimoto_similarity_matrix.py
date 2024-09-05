import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load your CSV file
df = pd.read_csv('molecules.smi') # the file should include only one column with "smiles" as column name

# Ensure there's a column named 'smiles' in your CSV
if 'smiles' not in df.columns:
    raise ValueError("CSV file must contain a 'smiles' column.")

# Convert SMILES to RDKit molecule objects
df['mol'] = df['smiles'].apply(Chem.MolFromSmiles)

# Drop any rows where the molecule could not be parsed
df = df.dropna(subset=['mol'])

# Compute Morgan fingerprints
df['fingerprint'] = df['mol'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=2048))

# Calculate the Tanimoto distance matrix (1 - similarity)
fingerprints = df['fingerprint'].values
n_molecules = len(fingerprints)
tanimoto_distance_matrix = np.zeros((n_molecules, n_molecules))

for i in range(n_molecules):
    for j in range(i, n_molecules):
        sim = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
        distance = 1 - sim  # Calculate distance
        tanimoto_distance_matrix[i, j] = distance
        tanimoto_distance_matrix[j, i] = distance

# Convert to a DataFrame with numerical indices
tanimoto_distance_df = pd.DataFrame(tanimoto_distance_matrix,
                                    columns=range(1, n_molecules + 1),
                                    index=range(1, n_molecules + 1))

# Save the matrix to a CSV file
tanimoto_distance_df.to_csv('tanimoto_distance_matrix.csv')

# Plot the Tanimoto distance matrix as a heatmap with darker colors for higher distances
plt.figure(figsize=(16, 14))
plt.imshow(tanimoto_distance_df.values[:2200, :2200], cmap='viridis', interpolation='bilinear')  #2200 equal the number of molecules
cbar = plt.colorbar()  # Get colorbar object
cbar.set_label('Tanimoto Distance', fontsize=16)  # Set font size for colorbar label

plt.xlabel('Molecule Index', fontsize=18)  # Increase font size for x-axis label
plt.ylabel('Molecule Index', fontsize=18)  # Increase font size for y-axis label

# Save the smoothened heatmap
plt.savefig('Tanimoto_Distance_Matrix.png', dpi=200, bbox_inches='tight')
