<div align="center">

# Health Data Science ML Pipeline  
### Powered by <br> Google AI Studio - <img src="https://upload.wikimedia.org/wikipedia/commons/b/b5/Google_ai_studio_logo.png" width="30"/> <br> Codex - <img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Openai.png" width="30"/>

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-brightgreen)]()
[![React](https://img.shields.io/badge/Frontend-React-blueviolet)]()
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange)]()

</div>

---

## Abstract

Atualmmente existem poucas ferramentas de modelagem preditiva acessíveis e amigáveis para profissionais de saúde de uma forma geral, ferramentas que permitam a este público explorar de forma fácil e rápida **Machine Learning** (aprendizado de máquina) em seus próprios bancos de dados clínicos sem depender de fluxos de trabalho complexos.  

O **Health Data Science ML Pipeline** foi criado para preencher essa lacuna — oferecendo um **workflow padronizado e automatizado** que guia o usuário desde o upload do dataset até a geração dos resultados e interpretação dos modelos.  

Este projeto foi desenvolvido com assistência do **Google AI Studio** e do **Codex da OpenAI**, combinando automação em análise de dados e suporte de IA para interpretação inteligente de resultados.

---

## Dataset Format (Input Specification)

Atualmente o programa exige datasets em formato **Excel (.xlsx)** conforme as regras abaixo:

- Primeira coluna: **ID exclusivo** (sem repetições).  
- Última coluna: **desfecho binário** (0 ou 1).  
- Colunas intermediárias: **variáveis independentes numéricas**.  
  - Variáveis categóricas devem **preferencialmente** estar previamente convertidas em formato *one-hot-encoded*.  

Exemplo simplificado:

| ID | idade | pressao | glicose | sexo_M | sexo_F | desfecho |
|----|--------|----------|----------|---------|---------|-----------|
| 1 | 67 | 132 | 95 | 1 | 0 | 1 |
| 2 | 74 | 141 | 103 | 0 | 1 | 0 |

---

## Workflow Overview

O fluxo de trabalho foi desenhado para que o profissional de saúde possa realizar análises de forma **intuitiva e reproduzível**:

1. **Upload do banco de dados**  
   O usuário envia um arquivo Excel conforme o formato exigido.

2. **Análise exploratória automática (EDA)**  
   O sistema exibe:
   - Porcentagem de desfecho (para checar balanceamento da amostra)  
   - Boxplots e histogramas para identificar outliers  
   - Tabelas de valores faltantes  
   - Correlações e distribuições das variáveis  

3. **Correção da base**  
   Com base nas sugestões automáticas (missing, outliers, checagem pareada de multicolinearidade), é possível limpar o dataset via interface.

4. **Seleção de modelos**  
   Usuário escolhe os algoritmos a testar (atualmente disponíveis apenas os mais básicos como **Logistic Regression, KNN, SVM, Random Forest, Gradient Boosting, K-Means**) e define alguns parâmetros (valor de K, Clusters e checagem de acotovelamento no K-MEANS). Em cada família de modelo, o programa testa de forma sequencial diversos hiperparametros para encontrar aquele com melhor desempenho (baseado em área sob a curva ROC e F1-score). 

5. **Treinamento automatizado**  
   São treinados modelos de machine learning, em seguido são geradas métricas como *AUC*, *sensibilidade/especificidade*, *F1-score*, *acurácia*, *ROC curves*, *confusion matrix* e *feature importances* para comparação.

6. **Resultados e download**  
   Relatórios e modelos otimizados podem ser baixados junto dos valores dos hiperparâmetros de melhor desempenho.

7. **Interpretação via IA (opcional)**  
   Se o usuário informar sua API Key, o **Google Gemini** fornece um resumo textual interpretando os achados clínicos dos modelos. A chave de API pode ser obtida gratuitamente em [Google AI Studio](https://aistudio.google.com/app/apikey)

---

## 🎥 Instructional Video

[![Watch the video](https://img.youtube.com/vi/_x5e5cBSl70/maxresdefault.jpg)](https://www.youtube.com/watch?v=_x5e5cBSl70)

Este tutorial em video demonstra:
- Introdução
- Fluxo de trabalho
- Como instalar o app localmente
- Como executar o programa e criar uma chave de API do Google
- Fazer upload do banco de dados
- Treinar e avaliar modelos de machine learning básicos (KNN, SVM, Logistic Regression, etc)
- Interpretar os resultados explorando a curva ROC, métricas e parâmetros


---

## Running Locally (Windows)

**Pré-requisitos:**  
- Python 3.11 ou superior já instalado (verificável com `python --version`)

**Passo-a-passo:**

1. **Baixe o repositório**  
   - Acesse o botão verde **"Code → Download ZIP"** no GitHub  
   - Descompacte o arquivo em uma pasta de sua preferência  

2. **Execute o Launcher**  
   - Dentro da pasta do projeto, localize o arquivo `launcher.py`  
   - Clique duas vezes nele
   - Será aberto o launcher de instalação, configuração de API e execução
   - Este launcher deve permanecer aberto até a finalização do programa  
   - Siga as instruções que aparecerem na tela  

O launcher cria automaticamente o ambiente virtual, instala dependências do **FastAPI (backend)** e do **React (frontend)**, e inicia ambos os serviços localmente:  
- Backend disponível em `http://localhost:8000`  
- Interface (Frontend) em `http://localhost:3000`

Então abre o navegador após 30 segundos dos servidores rodando já com a interface gráfica pronta para uso. Após o uso do programa, o usuário deve clicar em "End application" no laucher.


---

## Running in Google Colab (with LocalTunnel)

O programa também pode ser executado em um notebook Jupyter através do Google Colab (atualmmente apenas o server backend). Para isso, execute o backend diretamente em um notebook do Colab e conecte a interface local usando o túnel fornecido conforme o processo abaixo:

### 1. Clone o repositório

```
!git clone https://github.com/<your-account>/Basic-ML-for-Binary-Health-Outcomes.git
%cd Basic-ML-for-Binary-Health-Outcomes
```

### 2. Instale as dependências

```
!pip install -r backend/requirements.txt
!npm install -g localtunnel
```

> Se o comando `npm` não estiver disponível, execute:  
> `!apt-get install nodejs npm`

### 3. Inicie o backend com túnel público

```
!python backend/colab_runner.py --port 8000
```

A célula exibirá os endereços locais e públicos (via `loca.lt`). Mantenha a célula ativa para manter o servidor online.

**Parâmetros úteis:**
- `--no-localtunnel`: roda o backend apenas localmente  
- `--subdomain nome`: solicita um subdomínio específico  
- `--log-level debug`: aumenta a verbosidade do log  

### 4. Conecte o frontend (opcional)

No arquivo `.env.local` da sua máquina local, atualize:

```
VITE_API_URL=https://<seu-túnel>.loca.lt/api
```

Então rode:

```
npm run dev
```

Abra o URL impresso pelo Vite (geralmente `http://localhost:5173`) e teste o backend hospedado no Colab.

---

## Details: Backend and Frontend Overview

- **Frontend (React + Vite)**  
  Responsável pela interação do usuário, upload do dataset, visualização de gráficos e resultados de modelagem.  

  Comando para desenvolvimento:
  ```
  npm run dev
  ```

- **Backend (FastAPI + scikit-learn)**  
  Camada analítica que executa a EDA, limpeza, e treinamento dos modelos.  
  Comando para execução isolada:
  ```
  uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
  ```

---

## Details: Environment Variables (.env.local)

É necessário um arquivo `.env.local` na raiz do projeto com os valores abaixo:

```
VITE_API_URL=http://localhost:8000/api
GEMINI_API_KEY=your-gemini-api-key
```

- `VITE_API_URL`: endereço do backend (local ou remoto)  
- `GEMINI_API_KEY`: chave gratuita obtida em [Google AI Studio](https://aistudio.google.com/app/apikey)

---

## Quickstart Recap

1. **Rodar o backend (local ou Colab)**  
2. **Garantir que está acessível** (`/docs` do FastAPI)  
3. **Configurar `VITE_API_URL` corretamente**  
4. **Rodar frontend (`npm run dev`)**  
5. **Fazer upload do dataset e seguir o pipeline**  
6. **Usar interpretação Gemini se desejado**

---

<div align="center">
Feito com usando FastAPI, Vite e scikit-learn.  
Com apoio das APIs da Google AI e OpenAI Codex.
</div>
