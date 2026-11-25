# Esteira de Aprendizado de Máquina — Classificação de Vinhos

Trabalho da disciplina de Inteligência Artificial — Fatec Praia Grande (Análise e Desenvolvimento de Sistemas).

**Aluna:** Sophia Sugawara
**Vídeo de apresentação:** [https://youtu.be/JuerHFwSOr8]

## Descrição do projeto

Esteira de aprendizado de máquina que classifica vinhos a partir de sua análise química, prevendo de qual das 3 cultivares de uva (de uma mesma região da Itália) o vinho vem. Aplicação real: autenticação de origem e controle de qualidade na indústria de bebidas.

- **Base de dados:** [Wine — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/109/wine) (178 amostras, 13 variáveis químicas, 3 classes).
- **Ambiente de desenvolvimento:** Amazon SageMaker (AWS Academy), Python 3.

### Etapas da esteira (todas no notebook)

1. Carga da base
2. Estatísticas descritivas gerais
3. Transformações nas colunas (rótulo 0–2 e log em `prolina`; padronização dentro do pipeline)
4. Transformações nas linhas (duplicatas, outliers por IQR e embaralhamento)
5. Divisão estratificada em treino (60%), validação (20%) e teste (20%)
6. Treinamento de 3 modelos (Regressão Logística, KNN e Random Forest) e escolha pela validação
7. Avaliação final no teste: matriz de confusão, acurácia e relatório por classe (mais validação cruzada de 5 partições)
8. Salvamento do modelo (`modelo_vinho.joblib`) e predição de um vinho novo

### Resultados obtidos

- Modelo escolhido: Random Forest
- Acurácia no teste: 97,14% (34 acertos em 35 amostras)
- Validação cruzada (5 partições): média de 97,7%

## Arquivos do repositório

| Arquivo | Descrição |
|---|---|
| `esteira_ml_vinhos.ipynb` | Notebook com toda a esteira |
| `wine.data` | Base de dados (CSV sem cabeçalho) |
| `README.md` | Este arquivo |

O arquivo `modelo_vinho.joblib` é gerado pela execução do notebook (é um arquivo binário, não pode ser aberto como texto).

## Como reproduzir a execução apresentada no vídeo

### Opção A — Amazon SageMaker (AWS Academy), como no vídeo

1. Inicie o laboratório no AWS Academy e abra o console da AWS.
2. Acesse o **Amazon SageMaker** e crie (ou inicie) uma **instância de notebook**.
3. Abra o JupyterLab e envie para a pasta `SageMaker` os arquivos `esteira_ml_vinhos.ipynb` e `wine.data` (botão de upload do painel de arquivos).
4. Abra o notebook com o kernel `conda_python3`.
5. Execute tudo: menu **Kernel > Restart Kernel and Run All Cells**.
6. Ao final, o arquivo `modelo_vinho.joblib` aparece na mesma pasta.

### Opção B — Execução local

1. Instale o Python 3.9 ou superior.
2. Clone o repositório e entre na pasta:
   ```bash
   git clone https://github.com/sophia-sugawara/Esteira-Aprendizado-Vinhos.git
   cd Esteira-Aprendizado-Vinhos
   ```
3. Instale as dependências:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn joblib jupyter
   ```
4. Abra o notebook e execute todas as células:
   ```bash
   jupyter notebook esteira_ml_vinhos.ipynb
   ```

### Observações

- Se o arquivo `wine.data` não estiver na pasta, o notebook o baixa automaticamente do repositório da UCI (requer internet).
- O notebook usa semente fixa (`SEED = 42`), então os resultados devem ser os mesmos em cada execução.
- Para usar o modelo salvo fora do notebook:
  ```python
  import joblib
  modelo = joblib.load("modelo_vinho.joblib")
  ```
  Atenção: o modelo espera a coluna `prolina` já transformada com `np.log1p`, como é feito na função `prever_vinho` do notebook.
