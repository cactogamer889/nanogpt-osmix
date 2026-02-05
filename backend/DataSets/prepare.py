"""
Script para preparar dataset customizado para treinar osmix_base
Detecta e processa automaticamente TODOS os datasets em data/custom/
Suporta:
- CodeSearchNet (JavaScript, Java, Python)
- OASST1 (JSONL.gz e parquet)
- WikiText (parquet)
- Arquivos de texto (.txt) em qualquer subdiretório
- Arquivos JSONL
"""

import os
import json
import gzip
import zipfile
import tiktoken
import numpy as np
from pathlib import Path
from tqdm import tqdm
import tempfile

# Tentar importar pandas para parquet (opcional)
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("⚠️  pandas não instalado - datasets parquet serão ignorados")
    print("   Instale com: pip install pandas pyarrow")

# Configurações
input_dir = os.path.dirname(__file__)
output_dir = input_dir

# Encoding GPT-2 BPE
enc = tiktoken.get_encoding("gpt2")

# Linguagens do CodeSearchNet que queremos processar
CODESEARCHNET_LANGUAGES = ['javascript', 'java', 'python']

def tokenize_text(text):
    """Tokeniza texto e retorna lista de IDs"""
    if not text or len(text.strip()) == 0:
        return []
    ids = enc.encode_ordinary(text)
    ids.append(enc.eot_token)
    return ids

def prepare_from_text_file(input_file):
    """Prepara dataset de um único arquivo de texto"""
    print(f"Lendo arquivo: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = f.read()
    
    # Dividir em train/val (90/10)
    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]
    
    # Tokenizar
    print("Tokenizando dados de treino...")
    train_ids = tokenize_text(train_data)
    print("Tokenizando dados de validação...")
    val_ids = tokenize_text(val_data)
    
    return train_ids, val_ids

def prepare_from_codesearchnet(codesearchnet_dir):
    """Prepara dataset do CodeSearchNet - apenas JavaScript, Java e Python"""
    codesearchnet_path = os.path.join(codesearchnet_dir, 'code_search_net')
    data_dir = os.path.join(codesearchnet_path, 'data')
    
    if not os.path.exists(data_dir):
        return None, None
    
    print("=" * 60)
    print("Processando CodeSearchNet (JavaScript, Java, Python)")
    print("=" * 60)
    
    all_texts = []
    
    for lang in CODESEARCHNET_LANGUAGES:
        zip_path = os.path.join(data_dir, f"{lang}.zip")
        
        if not os.path.exists(zip_path):
            print(f"⚠️  {lang}.zip não encontrado, pulando...")
            continue
        
        print(f"\nProcessando {lang}...")
        
        # Extrair zip temporariamente
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Encontrar arquivos JSONL
            lang_dir = os.path.join(temp_dir, lang, 'final', 'jsonl')
            
            if not os.path.exists(lang_dir):
                print(f"⚠️  Estrutura não encontrada para {lang}")
                continue
            
            # Processar train, valid e test
            for split in ['train', 'valid', 'test']:
                split_dir = os.path.join(lang_dir, split)
                if not os.path.exists(split_dir):
                    continue
                
                jsonl_files = list(Path(split_dir).glob("*.jsonl.gz"))
                jsonl_files.extend(list(Path(split_dir).glob("*.jsonl")))
                
                for jsonl_file in tqdm(jsonl_files, desc=f"  {split}"):
                    try:
                        is_gzip = str(jsonl_file).endswith('.gz')
                        open_func = gzip.open if is_gzip else open
                        mode = 'rt' if is_gzip else 'r'
                        
                        with open_func(jsonl_file, mode, encoding='utf-8') as f:
                            for line in f:
                                try:
                                    data = json.loads(line.strip())
                                    code = data.get('whole_func_string', '') or data.get('original_string', '')
                                    if code and len(code.strip()) > 10:
                                        all_texts.append(code)
                                except (json.JSONDecodeError, KeyError):
                                    continue
                    except Exception as e:
                        print(f"    Erro: {e}")
                        continue
    
    if not all_texts:
        return None, None
    
    print(f"\nTotal de funções: {len(all_texts):,}")
    
    # Combinar e dividir
    combined_text = "\n\n".join(all_texts)
    n = len(combined_text)
    train_data = combined_text[:int(n*0.9)]
    val_data = combined_text[int(n*0.9):]
    
    # Tokenizar
    print("\nTokenizando...")
    train_ids = tokenize_text(train_data)
    val_ids = tokenize_text(val_data)
    
    return train_ids, val_ids

def prepare_from_oasst1(oasst1_dir):
    """Prepara dataset OASST1"""
    oasst1_path = os.path.join(oasst1_dir, 'oasst1')
    if not os.path.exists(oasst1_path):
        return None, None
    
    print("=" * 60)
    print("Processando OASST1")
    print("=" * 60)
    
    all_texts = []
    
    # Processar JSONL.gz files (prioridade para ready)
    jsonl_files = [
        os.path.join(oasst1_path, '2023-04-12_oasst_ready.messages.jsonl.gz'),
        os.path.join(oasst1_path, '2023-04-12_oasst_all.messages.jsonl.gz'),
    ]
    
    for jsonl_file in jsonl_files:
        if not os.path.exists(jsonl_file):
            continue
        
        print(f"Processando {os.path.basename(jsonl_file)}...")
        try:
            with gzip.open(jsonl_file, 'rt', encoding='utf-8') as f:
                for line in tqdm(f, desc="  Lendo linhas"):
                    try:
                        data = json.loads(line.strip())
                        # OASST1 tem estrutura de mensagens
                        text = data.get('text', '') or data.get('message', '')
                        if text and len(text.strip()) > 10:
                            all_texts.append(text)
                    except (json.JSONDecodeError, KeyError):
                        continue
        except Exception as e:
            print(f"  Erro: {e}")
            continue
    
    # Se não encontrou nada nos JSONL, tentar parquet
    if not all_texts and HAS_PANDAS:
        data_dir = os.path.join(oasst1_path, 'data')
        if os.path.exists(data_dir):
            parquet_files = list(Path(data_dir).glob("*.parquet"))
            for pq_file in tqdm(parquet_files, desc="Lendo parquet"):
                try:
                    df = pd.read_parquet(pq_file)
                    for col in ['text', 'message', 'content']:
                        if col in df.columns:
                            texts = df[col].dropna().astype(str)
                            all_texts.extend([t for t in texts if len(t.strip()) > 10])
                            break
                except Exception as e:
                    print(f"Erro ao ler {pq_file}: {e}")
    
    if not all_texts:
        return None, None
    
    print(f"Total de textos: {len(all_texts):,}")
    
    # Combinar e dividir
    combined_text = "\n\n".join(all_texts)
    n = len(combined_text)
    train_data = combined_text[:int(n*0.9)]
    val_data = combined_text[int(n*0.9):]
    
    # Tokenizar
    print("Tokenizando...")
    train_ids = tokenize_text(train_data)
    val_ids = tokenize_text(val_data)
    
    return train_ids, val_ids

def prepare_from_wikitext(wikitext_dir):
    """Prepara dataset WikiText"""
    if not HAS_PANDAS:
        return None, None
    
    wikitext_path = os.path.join(wikitext_dir, 'wikitext')
    if not os.path.exists(wikitext_path):
        return None, None
    
    print("=" * 60)
    print("Processando WikiText")
    print("=" * 60)
    
    # Prioridade: raw > v1, 103 > 2
    versions = ['wikitext-103-raw-v1', 'wikitext-2-raw-v1', 'wikitext-103-v1', 'wikitext-2-v1']
    
    all_texts = []
    
    for version in versions:
        version_dir = os.path.join(wikitext_path, version)
        if not os.path.exists(version_dir):
            continue
        
        print(f"Processando {version}...")
        
        # Processar arquivos parquet (train primeiro)
        train_files = sorted(Path(version_dir).glob("train*.parquet"))
        val_files = sorted(Path(version_dir).glob("validation*.parquet"))
        test_files = sorted(Path(version_dir).glob("test*.parquet"))
        
        all_parquet = train_files + val_files + test_files
        
        for pq_file in tqdm(all_parquet, desc=f"  {version}"):
            try:
                df = pd.read_parquet(pq_file)
                if 'text' in df.columns:
                    texts = df['text'].dropna().astype(str)
                    all_texts.extend([t for t in texts if len(t.strip()) > 10])
            except Exception as e:
                print(f"Erro ao ler {pq_file}: {e}")
        
        if all_texts:
            break  # Usar primeira versão encontrada
    
    if not all_texts:
        return None, None
    
    print(f"Total de textos: {len(all_texts):,}")
    
    # Combinar e dividir
    combined_text = "\n\n".join(all_texts)
    n = len(combined_text)
    train_data = combined_text[:int(n*0.9)]
    val_data = combined_text[int(n*0.9):]
    
    # Tokenizar
    print("Tokenizando...")
    train_ids = tokenize_text(train_data)
    val_ids = tokenize_text(val_data)
    
    return train_ids, val_ids

def combine_datasets(*dataset_lists):
    """Combina múltiplos datasets"""
    all_train_ids = []
    all_val_ids = []
    
    for train_ids, val_ids in dataset_lists:
        if train_ids is not None and val_ids is not None and len(train_ids) > 0:
            all_train_ids.extend(train_ids)
            all_val_ids.extend(val_ids)
    
    return all_train_ids, all_val_ids

if __name__ == '__main__':
    print("=" * 60)
    print("Preparando Dataset para osmix_base")
    print("=" * 60)
    print("\nDetectando datasets em:", input_dir)
    print()
    
    datasets_processed = []
    dataset_names = []
    
    # 1. CodeSearchNet
    train_ids, val_ids = prepare_from_codesearchnet(input_dir)
    if train_ids:
        datasets_processed.append((train_ids, val_ids))
        dataset_names.append("CodeSearchNet")
        print(f"✓ CodeSearchNet: {len(train_ids):,} train tokens, {len(val_ids):,} val tokens\n")
    
    # 2. OASST1
    train_ids, val_ids = prepare_from_oasst1(input_dir)
    if train_ids:
        datasets_processed.append((train_ids, val_ids))
        dataset_names.append("OASST1")
        print(f"✓ OASST1: {len(train_ids):,} train tokens, {len(val_ids):,} val tokens\n")
    
    # 3. WikiText
    train_ids, val_ids = prepare_from_wikitext(input_dir)
    if train_ids:
        datasets_processed.append((train_ids, val_ids))
        dataset_names.append("WikiText")
        print(f"✓ WikiText: {len(train_ids):,} train tokens, {len(val_ids):,} val tokens\n")
    
    # 4. Procurar input.txt em subdiretórios (char, shake, etc)
    for subdir in Path(input_dir).iterdir():
        if subdir.is_dir() and subdir.name not in ['code_search_net', 'oasst1', 'wikitext', '.git']:
            sub_input = subdir / 'input.txt'
            if sub_input.exists():
                print(f"Processando {subdir.name}/input.txt...")
                train_ids, val_ids = prepare_from_text_file(str(sub_input))
                if train_ids:
                    datasets_processed.append((train_ids, val_ids))
                    dataset_names.append(subdir.name)
                    print(f"✓ {subdir.name}: {len(train_ids):,} train tokens, {len(val_ids):,} val tokens\n")
    
    # 5. Arquivo input.txt na raiz
    input_file = os.path.join(input_dir, 'input.txt')
    if os.path.exists(input_file):
        train_ids, val_ids = prepare_from_text_file(input_file)
        if train_ids:
            datasets_processed.append((train_ids, val_ids))
            dataset_names.append("input.txt")
            print(f"✓ input.txt: {len(train_ids):,} train tokens, {len(val_ids):,} val tokens\n")
    
    # 6. Arquivo input.jsonl na raiz
    jsonl_file = os.path.join(input_dir, 'input.jsonl')
    if os.path.exists(jsonl_file):
        print("Processando input.jsonl...")
        # Processar JSONL
        all_texts = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Lendo linhas"):
                try:
                    data = json.loads(line.strip())
                    text = (data.get('text') or data.get('content') or 
                           data.get('message') or '')
                    if text and len(str(text).strip()) > 10:
                        all_texts.append(str(text))
                except (json.JSONDecodeError, KeyError):
                    continue
        
        if all_texts:
            combined_text = "\n\n".join(all_texts)
            n = len(combined_text)
            train_data = combined_text[:int(n*0.9)]
            val_data = combined_text[int(n*0.9):]
            train_ids = tokenize_text(train_data)
            val_ids = tokenize_text(val_data)
            if train_ids:
                datasets_processed.append((train_ids, val_ids))
                dataset_names.append("input.jsonl")
                print(f"✓ input.jsonl: {len(train_ids):,} train tokens, {len(val_ids):,} val tokens\n")
    
    # Combinar todos
    if not datasets_processed:
        print("\n❌ Nenhum dataset encontrado!")
        print("\nDatasets suportados:")
        print("  - CodeSearchNet: data/custom/code_search_net/data/*.zip")
        print("  - OASST1: data/custom/oasst1/")
        print("  - WikiText: data/custom/wikitext/")
        print("  - Arquivo texto: data/custom/input.txt ou data/custom/*/input.txt")
        print("  - Arquivo JSONL: data/custom/input.jsonl")
        exit(1)
    
    print("=" * 60)
    print("Combinando todos os datasets...")
    print("=" * 60)
    
    train_ids, val_ids = combine_datasets(*datasets_processed)
    
    # Estatísticas finais
    print(f"\n{'='*60}")
    print("Estatísticas Finais do Dataset")
    print(f"{'='*60}")
    print(f"  Datasets processados: {', '.join(dataset_names)}")
    print(f"  Total de datasets: {len(datasets_processed)}")
    print(f"  Train tokens: {len(train_ids):,}")
    print(f"  Val tokens: {len(val_ids):,}")
    print(f"  Total tokens: {len(train_ids) + len(val_ids):,}")
    
    # Salvar arquivos binários
    train_path = os.path.join(output_dir, 'train.bin')
    val_path = os.path.join(output_dir, 'val.bin')
    
    print(f"\nSalvando {train_path}...")
    train_ids_array = np.array(train_ids, dtype=np.uint16)
    train_ids_array.tofile(train_path)
    
    print(f"Salvando {val_path}...")
    val_ids_array = np.array(val_ids, dtype=np.uint16)
    val_ids_array.tofile(val_path)
    
    print(f"\n{'='*60}")
    print("✓ Dataset preparado com sucesso!")
    print(f"{'='*60}")
    print(f"  Arquivos salvos em: {output_dir}")
    print(f"  - train.bin ({len(train_ids):,} tokens)")
    print(f"  - val.bin ({len(val_ids):,} tokens)")
