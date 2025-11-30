import os
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
import torch

import scipy.sparse as sp

from transformers import AutoTokenizer, AutoModel

import scanpy as sc


def np_norm(inp):
    max_in = np.max(inp)
    min_in = np.min(inp)
    return (inp - min_in) / (max_in - min_in + 1e-8)


def gray_value_of_gene(gene_class, gene_order):
    gene_order = list(gene_order)
    Index = gene_order.index(gene_class)
    interval = 255 / len(gene_order)
    value = Index * interval
    return int(value)


def load_data(data_root, dataset_use, status, SR_times, gene_num, all_gene,
              gene_order=None, gene_name_order=None, pre_model_dir=None):
    dataset = Data_all(
        data_root, dataset_use, SR_times, status, gene_num, all_gene,
        gene_order, gene_name_order, pre_model_dir=pre_model_dir
    )
    return dataset


class Data_all(Dataset):
    

    _ALL_CLASSES = [
        'Human_breast_cancer',
        'Human_colon_cancer',
        'Human_healthy_kidney',
        'Human_kidney_cancer',
        'Mouse_healthy_brain'
    ]

    def __init__(self, data_root, dataset_use, SR_times, status, gene_num,
                 all_gene, gene_order, gene_name_order, pre_model_dir):
        self.data_root = data_root
        self.status = status
        self.gene_num = int(gene_num) if gene_num is not None else 20
        self.selected_classes = self._ALL_CLASSES

        self.selected_patches = []
        for cls_name in self.selected_classes:
            print(f"🔹 Scanning class: {cls_name}")
            cls_patches = self._load_patch_list(self.data_root, cls_name)
            # (cls_name, sub, layer_name, patch_id)
            self.selected_patches.extend([(cls_name, *p) for p in cls_patches])

        if status == 'Test':
            self._sort_patches()

        self.sc_reference = {}
        for cls_name in self.selected_classes:
            self.sc_reference[cls_name] = self._load_sc_reference(cls_name)

        self.sub_gene_info = self._load_subdataset_gene_info()

        self._load_coexpression(self.data_root)

        
        self._load_data(self.data_root)

        self._normalize_data()

        self._load_bert_model(pre_model_dir)
        self._embed_metadata()

        self._precompute_gene_index_maps()

        print(f"[{status}] Loaded {len(self.selected_patches)} patches from {len(self.selected_classes)} classes.")

    def _load_patch_list(self, data_root, cls_name):
        patches = []
        root = os.path.join(data_root, cls_name)
        if not os.path.isdir(root):
            print(f"⚠️ Missing class dir: {root}")
            return patches

        for sub in sorted(os.listdir(root)):
            sub_path = os.path.join(root, sub)
            if not os.path.isdir(sub_path) or sub == 'sc':
                continue
            extract_base = os.path.join(sub_path, 'HR_ST', 'extract')
            if not os.path.isdir(extract_base):
                continue
            for layer_name in sorted(os.listdir(extract_base)):
                layer_dir = os.path.join(extract_base, layer_name)
                if not os.path.isdir(layer_dir):
                    continue
                for patch_id in sorted(os.listdir(layer_dir)):
                    patches.append((sub, layer_name, patch_id))
        return patches

    def _sort_patches(self):
        # selected_patches: (cls_name, sub, layer, patch_id)
        self.selected_patches.sort(key=lambda x: (x[0], x[1], x[2], x[3]))

    def _load_sc_reference(self, cls_name):
        sc_dir = os.path.join(self.data_root, cls_name, 'sc')
        if not os.path.isdir(sc_dir):
            print(f"⚠️ Missing sc dir for {cls_name}: {sc_dir}")
            return None

        h5_list = [f for f in os.listdir(sc_dir) if f.endswith('.h5ad')]
        if len(h5_list) != 1:
            raise RuntimeError(f"{cls_name}/sc 必须且仅包含一个 .h5ad 文件，当前: {h5_list}")

        h5_path = os.path.join(sc_dir, h5_list[0])
        print(f"🔹 Loading scRNA data for {cls_name}: {h5_path}")

        adata = sc.read_h5ad(h5_path)
        adata.var_names = adata.var_names.astype(str)
        adata.var_names_make_unique()

        max_genes = 100
        keep_n_genes = min(max_genes, adata.n_vars)
        adata = adata[:, adata.var_names[:keep_n_genes]].copy()

        import numpy as np
        from scipy import sparse

        X = adata.X
        if sparse.issparse(X):
            cell_counts = np.array(X.sum(axis=1)).ravel()
        else:
            cell_counts = X.sum(axis=1)

        nonzero_mask = cell_counts > 0
        if nonzero_mask.sum() < adata.n_obs:
            adata = adata[nonzero_mask, :].copy()

        sc.pp.normalize_total(adata, inplace=True)
        sc.pp.log1p(adata)
        max_cells = 100
        if adata.n_obs > max_cells:
            rng = np.random.RandomState(0)
            keep_idx = rng.choice(adata.n_obs, size=max_cells, replace=False)
            adata = adata[keep_idx, :].copy()

        print(f"✅ {cls_name} scRNA loaded: cells={adata.n_obs}, genes={adata.n_vars}")
        return adata

    def _load_subdataset_gene_info(self):
        info = {}
        for cls_name in self.selected_classes:
            cls_root = os.path.join(self.data_root, cls_name)
            if not os.path.isdir(cls_root):
                continue
            for sub in sorted(os.listdir(cls_root)):
                sub_path = os.path.join(cls_root, sub)
                if not os.path.isdir(sub_path) or sub == 'sc':
                    continue
                order_path = os.path.join(sub_path, 'gene_order.npy')
                name_path = os.path.join(sub_path, 'gene_name.txt')
                if not (os.path.exists(order_path) and os.path.exists(name_path)):
                    print(f"⚠️ Missing gene files for {(cls_name, sub)}: {order_path} | {name_path}")
                    continue
                order = np.load(order_path)[:self.gene_num].astype(int)
                names = np.loadtxt(name_path, dtype=str)[:self.gene_num]
                info[(cls_name, sub)] = {"idxN": order, "namesN": names.tolist()}
        return info

    def _load_coexpression(self, root):
        self.co_expression_dict = {}
        for cls_name in self.selected_classes:
            coex_path = os.path.join(root, cls_name, 'coexpression_matrix.npy')
            if not os.path.exists(coex_path):
                coex_path = os.path.join(root, cls_name, 'gene_coexpre.npy')
            if os.path.exists(coex_path):
                co_expression = np.load(coex_path)
                keep = min(co_expression.shape[0], self.gene_num * 10)
                self.co_expression_dict[cls_name] = np.asarray(co_expression[:keep, :keep], dtype=np.float32)

    def _load_data(self, root):
        sr_list, spot_list, wsi5120_list, wsi320_list, wsimask_list, sc_list = [], [], [], [], [], []
        scgpt_list, prehe_list = [], []

        for cls_name, sub, layer_name, patch_id in self.selected_patches:
            base = os.path.join(root, cls_name, sub)
            key = (cls_name, sub)
            if key not in self.sub_gene_info:
                print(f"⚠️ Skip patch without gene info: {(cls_name, sub, layer_name, patch_id)}")
                continue

            idxN = self.sub_gene_info[key]['idxN']

            paths = {
                'hr': os.path.join(base, 'HR_ST', 'extract', layer_name, patch_id, 'HR_ST_256.npz'),
                'spot': os.path.join(base, 'spot_ST', 'extract', layer_name, patch_id, 'spot_ST.npz'),
                'wsi5120': os.path.join(base, 'WSI', 'extract', layer_name, patch_id, '5120_to256.npy'),
                'wsi320': os.path.join(base, 'WSI', 'extract', layer_name, patch_id, '320_to16.npy'),
                'mask': os.path.join(base, 'WSI', 'extract', layer_name, patch_id, 'cell_mask.npy'),
            }

            if not (os.path.exists(paths['hr']) and os.path.exists(paths['spot']) and
                    os.path.exists(paths['wsi5120']) and os.path.exists(paths['wsi320']) and
                    os.path.exists(paths['mask'])):
                print(f"⚠️ Missing files for patch: {paths}")
                continue

          
            sr = sp.load_npz(paths['hr'])[:, idxN].toarray().reshape(256, 256, -1).transpose(2, 0, 1)
            spot = sp.load_npz(paths['spot'])[:, idxN].toarray().reshape(26, 26, -1).transpose(2, 0, 1)

            wsi5120 = np.load(paths['wsi5120']).transpose(2, 0, 1)
            wsi320 = np.load(paths['wsi320']).transpose(0, 3, 1, 2)
            wsimask = np.load(paths['mask']).transpose(2, 0, 1)

            sr_list.append(sr.astype(np.float32))
            spot_list.append(spot.astype(np.float32))
            wsi5120_list.append(wsi5120.astype(np.float32))
            wsi320_list.append(wsi320.astype(np.float32))
            wsimask_list.append(wsimask.astype(np.float32))

            sc_list.append(self.sc_reference.get(cls_name, None))

        
            scgpt_dir = os.path.join(base, 'spot_ST', 'extract', layer_name, patch_id, 'scgpt_data')
            gene_names = self.sub_gene_info[key]['namesN']
            scgpt_embed = load_scgpt_embedding(scgpt_dir, gene_names)
            scgpt_list.append(scgpt_embed)

            pre_he_path = os.path.join(base, 'WSI', 'extract', layer_name, patch_id, 'pre_he.npy')
            pre_he = load_pre_he(pre_he_path)
            if pre_he is None:
  
                print(f"⚠️ Missing pre_he for patch: {pre_he_path}, using zero array.")
            prehe_list.append(pre_he)
    
        self.SR_ST_all = np.asarray(sr_list, dtype=np.float32)
        self.spot_ST_all = np.asarray(spot_list, dtype=np.float32)
        self.WSI_5120_all = np.asarray(wsi5120_list, dtype=np.float32)
        self.WSI_320_all = np.asarray(wsi320_list, dtype=np.float32)
        self.WSI_mask_all = np.asarray(wsimask_list, dtype=np.float32)
        self.sc_all = sc_list
        self.scgpt_list = scgpt_list
        self.prehe_list = prehe_list

        if len(self.SR_ST_all) == 0:
            print("⚠️ No valid patches loaded. Please check directory structure and files.")

    
    def _normalize_data(self):
        for i in range(self.spot_ST_all.shape[0]):
            data = self.spot_ST_all[i]
            mins = data.min(axis=(1, 2), keepdims=True)
            maxs = data.max(axis=(1, 2), keepdims=True)
            denom = maxs - mins + 1e-8
            self.spot_ST_all[i] = (data - mins) / denom
        patch_max = np.max(self.SR_ST_all, axis=(1, 2, 3))
        patch_max[patch_max == 0] = 1.0
        self.patch_scale = patch_max.astype(np.float32)
        self.SR_ST_all = self.SR_ST_all / patch_max[:, None, None, None]


    def _load_bert_model(self, pre_model_dir):
        self.tokenizer = AutoTokenizer.from_pretrained(f'./{pre_model_dir}/bert', trust_remote_code=True)
        self.model = AutoModel.from_pretrained(f'./{pre_model_dir}/bert', local_files_only=True, trust_remote_code=True)
        self.model.eval()

    def _embed_metadata(self):
        prompt = ("Provide spatial transcriptomics data from the Xenium5k platform "
                  "for mouse species, with a cancer condition, and brain tissue type.")
        with torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors='pt')
            outputs = self.model(**inputs)
            self.metadata_feature = outputs.pooler_output.squeeze(0)

    def _precompute_gene_index_maps(self):
        N = len(self.SR_ST_all)
        self.gene_index_maps_all = np.zeros((N, self.gene_num, 256, 256), dtype=np.float32)
        for i in range(N):
            for j in range(self.gene_num):
                self.gene_index_maps_all[i, j] = np.full((256, 256), j / self.gene_num, dtype=np.float32)

    def __len__(self):
        return len(self.SR_ST_all)

    def __getitem__(self, idx):
        cls_name = self.selected_patches[idx][0]
        co_expression = torch.tensor(self.co_expression_dict[cls_name], dtype=torch.float32)

        patch_scale = torch.tensor(self.patch_scale[idx], dtype=torch.float32)
        sc_adata = self.sc_all[idx]          
        x = sc_adata.X                       
        if sp.issparse(x):
            x = x.toarray()
        sc_item = torch.as_tensor(x, dtype=torch.float32)

        return (
            self.SR_ST_all[idx],
            self.spot_ST_all[idx],
            self.WSI_5120_all[idx],
            self.WSI_320_all[idx],
            np.arange(self.gene_num, dtype=np.float32),
            self.gene_index_maps_all[idx],
            self.metadata_feature,
            patch_scale,
            co_expression,
            self.WSI_mask_all[idx],
            sc_item,
            self.scgpt_list[idx],
            self.prehe_list[idx]
        )

def load_scgpt_embedding(scgpt_dir, gene_names):
    G = len(gene_names)
    chunks = []

    for s in range(1, G+1, 5):   # group_size=5
        e = min(s+4, G)
        f = os.path.join(scgpt_dir, f"{s}to{e}.npy")
        if os.path.exists(f):
            chunks.append(np.load(f))
        else:
            chunks.append(np.zeros_like(chunks[-1]) if chunks else None)

    scgpt_full = np.concatenate(chunks, axis=0)
    return scgpt_full.astype(np.float32)


def load_pre_he(prehe_path):

    if os.path.exists(prehe_path):
        return np.load(prehe_path).astype(np.float32)
    else:
        return None
