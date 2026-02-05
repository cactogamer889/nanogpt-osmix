# Dataset Customizado para osmix_base

Este diretório é para preparar seu próprio dataset para treinar o modelo osmix_base (500M parâmetros).

## Formatos Suportados

O script `prepare.py` aceita três formatos:

### 1. Arquivo único de texto (`input.txt`)

Coloque um arquivo `input.txt` neste diretório com todo o texto que você quer usar para treinar.

```
data/custom/
  └── input.txt
```

### 2. Múltiplos arquivos de texto

Coloque vários arquivos `.txt` neste diretório. Todos serão combinados.

```
data/custom/
  ├── texto1.txt
  ├── texto2.txt
  └── texto3.txt
```

### 3. Arquivo JSONL (`input.jsonl`)

Arquivo JSONL onde cada linha é um JSON com campo `text` ou `content`:

```json
{"text": "Primeiro texto para treinar..."}
{"text": "Segundo texto para treinar..."}
{"content": "Terceiro texto..."}
```

## Como Usar

1. Coloque seus dados em um dos formatos acima neste diretório
2. Execute o script de preparação:

```bash
python data/custom/prepare.py
```

3. Isso vai criar `train.bin` e `val.bin` prontos para treinar

## Sugestões de Datasets

Para treinar um modelo de 500M, você precisa de bastante texto. Algumas opções:

### Opções Menores (GBs, não 40GB):

1. **WikiText-2** (~4MB texto, ~2M tokens)
   - Pequeno demais para 500M, mas bom para testar

2. **BookCorpus** (~5GB texto)
   - Livros diversos, bom para text generation

3. **C4 (Colossal Clean Crawled Corpus)** - versão filtrada
   - Pode baixar uma amostra menor

4. **Seu próprio dataset**
   - Combine textos de várias fontes
   - Artigos, livros, documentos, etc.

### Tamanho Recomendado

Para um modelo de 500M, idealmente você quer:
- **Mínimo**: ~1-2GB de texto (~500M-1B tokens)
- **Ideal**: ~5-10GB de texto (~2-5B tokens)
- **Máximo**: ~20GB de texto (~10B tokens)

Mais dados = melhor modelo, mas também mais tempo de treinamento.

## Exemplo: Criar dataset de múltiplos arquivos

```bash
# Criar diretório
mkdir -p data/custom

# Copiar seus arquivos de texto
cp meus_textos/*.txt data/custom/

# Preparar dataset
python data/custom/prepare.py
```

## Exemplo: Usar arquivo único

```bash
# Copiar seu arquivo
cp meu_texto_grande.txt data/custom/input.txt

# Preparar dataset
python data/custom/prepare.py
```
