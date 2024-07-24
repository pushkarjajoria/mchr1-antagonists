import torch
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd


class SMILESDataset(Dataset):
    def __init__(self, data_frame, smiles_col, target_col, feature_cols=None, transform=None):
        """
        Args:
            data_frame (pd.DataFrame): DataFrame containing the data.
            smiles_col (str): Column name for SMILES strings.
            target_col (str): Column name for target values.
            feature_cols (list of str, optional): List of additional feature columns.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_frame = data_frame
        self.smiles_col = smiles_col
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        smiles = self.data_frame.iloc[idx][self.smiles_col]
        target = self.data_frame.iloc[idx][self.target_col]

        if self.feature_cols:
            features = self.data_frame.iloc[idx][self.feature_cols].values
            sample = {'smiles': smiles, 'features': features, 'target': target}
        else:
            sample = {'smiles': smiles, 'target': target}

        if self.transform:
            sample = self.transform(sample)

        return sample


class FingerprintTransform:
    def __init__(self, radius=2, n_bits=2048):
        self.radius = radius
        self.n_bits = n_bits

    def __call__(self, sample):
        mol = Chem.MolFromSmiles(sample['smiles'])
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
        fp_array = torch.tensor(fp, dtype=torch.float32)

        if 'features' in sample:
            features = torch.tensor(sample['features'], dtype=torch.float32)
            return {'features': torch.cat((fp_array, features)),
                    'target': torch.tensor(sample['target'], dtype=torch.float32)}
        else:
            return {'features': fp_array, 'target': torch.tensor(sample['target'], dtype=torch.float32)}


# Replace 'your_file.sdf' with your SDF file path
sdf_file = 'data/MCHR1_patent.sdf'

# Create a SDF reader
supplier = Chem.SDMolSupplier(sdf_file)

# Initialize lists to hold the data
smiles_list = []
acvalues = []

acvalue_types = []
# Iterate over molecules in the SDF file
for i, mol in enumerate(supplier):
    if mol is not None:
        smiles = Chem.MolToSmiles(mol)
        properties = mol.GetPropsAsDict()

        # Add SMILES and properties to the lists
        # smiles_list.append(smiles)
        try:
            acval = int(properties['acvalue_uM'].split("\n")[0])
        except:
            print(f"Unable to process the acvalue for: {properties['acvalue_uM']}")
        # acvalues.append(acval)


# Create a DataFrame from the lists
df = pd.DataFrame(acvalues)
df.insert(0, 'SMILES', smiles_list)  # Insert SMILES column at the beginning

print(df)
