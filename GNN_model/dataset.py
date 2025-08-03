import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from rdkit import Chem, RDLogger
import warnings

warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

# ===================== FEATURE EXTRACTION MODULE =====================

def get_atom_features(atom):
    """
    Extract enhanced atomic features from RDKit atom object.

    Args:
        atom: RDKit atom object

    Returns:
        list: A list containing 25 atomic features
    """
    features = []

    # Basic atomic properties
    features.append(atom.GetAtomicNum())  # Atomic number
    features.append(atom.GetDegree())  # Number of bonds
    features.append(atom.GetFormalCharge())  # Formal charge
    features.append(int(atom.GetHybridization()))  # Hybridization type
    features.append(int(atom.GetIsAromatic()))  # Aromaticity flag
    features.append(atom.GetMass())  # Atomic mass

    # Electronic properties
    features.append(atom.GetTotalNumHs())  # Total hydrogen count (including implicit)
    features.append(atom.GetNumRadicalElectrons())  # Number of radical electrons
    features.append(int(atom.IsInRing()))  # Ring membership flag
    features.append(int(atom.IsInRingSize(3)))  # 3-membered ring flag
    features.append(int(atom.IsInRingSize(4)))  # 4-membered ring flag
    features.append(int(atom.IsInRingSize(5)))  # 5-membered ring flag
    features.append(int(atom.IsInRingSize(6)))  # 6-membered ring flag

    # Chemical environment features
    features.append(atom.GetTotalValence())  # Total valence
    features.append(atom.GetExplicitValence())  # Explicit valence

    # One-hot encoding for common atom types
    atom_types = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
    atom_symbol = atom.GetSymbol()
    for at in atom_types:
        features.append(int(atom_symbol == at))

    # Chirality information
    features.append(int(atom.GetChiralTag()))  # Chiral tag

    return features


def get_bond_features(bond):
    """
    Extract enhanced bond features from RDKit bond object.

    Args:
        bond: RDKit bond object

    Returns:
        list: A list containing 10 bond features
    """
    features = []

    # Basic bond properties
    features.append(int(bond.GetBondType()))  # Bond type
    features.append(int(bond.GetIsConjugated()))  # Conjugation flag
    features.append(int(bond.IsInRing()))  # Ring membership flag

    # Stereochemistry
    features.append(int(bond.GetStereo()))  # Stereochemistry type

    # One-hot encoding for bond types
    bond_types = [Chem.rdchem.BondType.SINGLE,
                  Chem.rdchem.BondType.DOUBLE,
                  Chem.rdchem.BondType.TRIPLE,
                  Chem.rdchem.BondType.AROMATIC]
    for bt in bond_types:
        features.append(int(bond.GetBondType() == bt))

    # Ring size information
    if bond.IsInRing():
        ring_info = bond.GetOwningMol().GetRingInfo()
        min_ring_size = min([len(ring) for ring in ring_info.AtomRings()
                             if bond.GetBeginAtomIdx() in ring and bond.GetEndAtomIdx() in ring],
                            default=0)
        features.append(min_ring_size)
    else:
        features.append(0)

    # Bond order
    features.append(int(bond.GetBondTypeAsDouble()))

    return features


def smiles_to_graph(smiles):
    """
    Convert SMILES string to PyTorch Geometric graph data object.

    Args:
        smiles (str): SMILES representation of molecule

    Returns:
        Data: PyTorch Geometric Data object or None if conversion fails
    """
    if pd.isna(smiles) or smiles == '':
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Extract node features (atoms)
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))

    x = torch.tensor(atom_features, dtype=torch.float)

    # Extract edge features and indices (bonds)
    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # Add both directions for undirected graph
        edge_indices.extend([[i, j], [j, i]])
        bond_features = get_bond_features(bond)
        edge_attrs.extend([bond_features, bond_features])

    if len(edge_indices) > 0:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        # Handle molecules with no bonds (single atoms)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 10), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# ===================== DATASET DEFINITION =====================

class MixtureDataset(Dataset):
    """
    Dataset class for molecular mixtures.

    Handles mixtures with 1-3 components, each with associated ratios.
    """

    def __init__(self, df, label):
        """
        Initialize mixture dataset.

        Args:
            df (pd.DataFrame): DataFrame containing mixture data
            label (str): Target property column name
        """
        self.data = []
        for _, row in df.iterrows():
            components = []
            ratios = []

            # Process component A
            if pd.notna(row['SMILES_A']) and row['Ratio_A'] > 0:
                graph_a = smiles_to_graph(row['SMILES_A'])
                if graph_a is not None:
                    components.append(graph_a)
                    ratios.append(row['Ratio_A'])

            # Process component B
            if pd.notna(row.get('SMILES_B', '')) and row.get('Ratio_B', 0) > 0:
                graph_b = smiles_to_graph(row['SMILES_B'])
                if graph_b is not None:
                    components.append(graph_b)
                    ratios.append(row['Ratio_B'])

            # Process component C
            if pd.notna(row.get('SMILES_C', '')) and row.get('Ratio_C', 0) > 0:
                graph_c = smiles_to_graph(row['SMILES_C'])
                if graph_c is not None:
                    components.append(graph_c)
                    ratios.append(row['Ratio_C'])

            if len(components) > 0:
                self.data.append({
                    'components': components,
                    'ratios': torch.tensor(ratios, dtype=torch.float),
                    'target': torch.tensor([row[label]], dtype=torch.float)
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    """
    Custom collate function for batching mixture data.

    Args:
        batch: List of mixture samples

    Returns:
        dict: Batched data dictionary
    """
    components_list = []
    ratios_list = []
    targets = []
    batch_indices = []

    for i, item in enumerate(batch):
        for j, comp in enumerate(item['components']):
            components_list.append(comp)
            batch_indices.append(i)
        ratios_list.append(item['ratios'])
        targets.append(item['target'])

    # Batch all molecular graphs
    batched_graphs = Batch.from_data_list(components_list) if components_list else None

    return {
        'graphs': batched_graphs,
        'ratios': ratios_list,
        'targets': torch.stack(targets),
        'batch_indices': batch_indices
    }
