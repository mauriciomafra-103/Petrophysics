# Bibliotecas mais comuns
import openpyxl

# Bibliotecas para leitura e processamentos dos dados
import pandas as pd
import numpy as np

# Bibliotecas para Processamento RMN
from sklearn import preprocessing
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error


def ObtencaoDadosNiumag(Diretorio_pasta, Arquivo_niumag, Inicio_conversao, Pontos_inversao,
                        Relaxacao = False, Distribuicao = False, T2_niumag = False, Erro = False,
                        Amplitudes = False, Poco = False, N_poco_i = 4, N_poco_f = 0, N_amp = 3):


    """
      Esta função trata o dados brutos de RMN que o software da Niumag exporta retornando um pandas.DataFrame com as informações
      relevantes e com aspecto mais legível.

      Args:
          Diretorio_pasta (str): O caminho do diretório onde está o arquivo excel exportado pelo software Niumag.
          Arquivo_niumag (str.xlsx): Nome do arquivo que está dentro do diretório e que é um arquivo tipo Excel.
          Inicio_conversao (int): Primeiro ponto de aquisição dos dados no arquivo Excel.
          Pontos_inversao (int): Quantos pontos da inversão foram usados.
          Relaxacao (bool) : Caso o usuário queira retornar os valores do Sinal e Distribuição T2.
          Distribuicao (bool): Caso o usuário queira retornar os valores do Sinal, Fiiting e Tempo de Relaxação.
          T2_niumag (bool): Caso o usuário queira retornar o processamento da média geométrica T2.
          Erro (bool): Caso o usuário deseje o Erro Fiiting.
          Amplitudes (bool): Caso o usuário deseje os picos das amplitudes.
          Poco (bool): Caso o usuário tenha as informações dos poços contidas no nome das amostras.
          N_poco_i (int): Identificação do começo do nome do poço.
          N_poco_f (int): Identificação do final do nome do poço.
          N_amp (int): Identificação da quantidade de amplitudes desejadas.

      Returns:
          pandas.DataFrame: Retorna um DataFrame com dados processados do Excel exportado pelo Software Niumag.

      Exemplos de Uso:
          Caso o usuário tenha um arquivo no estilo exportado pela Niumag que informa a amostra (no nome da amostra informa o nome do poço) essa função retornará os dados
          já específico para o processamento necessário.
      """

    niumag = str(Diretorio_pasta) + str(Arquivo_niumag)                           # Pasta do arquivp
    dados_niumag = pd.read_excel(niumag).drop('File Name', axis=1)                # Dataframe dos Dados da Niumag

    inicio = Inicio_conversao-2                                                   # Linha que se inicia os dados de da inversão
    final = inicio+Pontos_inversao                                                # Linha final da Inversão

    amostras = []
    poc = []
    tempo_relaxacao = []
    amplitude_sinal_niumag = []
    amplitude_sinal_fitting = []
    tempo_distribuicao = []
    distribuicao_t2 = []
    t2gm_niumag = []
    t2av_niumag = []
    fitting_erro = []
    amplitudes = []

    for i in np.arange(int(len(dados_niumag.columns)/7)):
      df = dados_niumag.T.reset_index().drop('index', axis = 1).T

      nome = dados_niumag.columns[i*7][:9]
      amostras.append(nome)

      if Poco == True:
        p = nome[N_poco_f:N_poco]
        poc.append(p)

      if Relaxacao == True:
        time = df[i*7][13:]
        sinal_niumag = df[i*7+1][13:]
        sinal_fiting = df[i*7+2][13:]
        tempo_relaxacao.append(time)
        amplitude_sinal_niumag.append(sinal_niumag)
        amplitude_sinal_fitting.append(sinal_fiting)

      if Distribuicao == True:
        tempo = df[i*7+3][inicio:final]
        dist = df[i*7+4][inicio:final]
        tempo_distribuicao.append(tempo)
        distribuicao_t2.append(dist)

      if T2_niumag == True:
        gm = float(df[i*7+2][1][7:-4])
        av = float(df[i*7+2][2][7:-4])
        t2gm_niumag.append(gm)
        t2av_niumag.append(av)

      if Erro == True:
        fit_erro = float(df[i*7][0][-5:])
        fitting_erro.append(fit_erro)

      if Amplitudes == True:
        amp = df[i*7+2][5:10].fillna(0).nlargest(N_amp)
        amplitudes.append(amp)

    df = pd.DataFrame({'Amostra': amostras})

    if Poco == True:
      df['Poço'] = poc

    if Relaxacao == True:
      df['Tempo Relaxacao'] = pd.Series(tempo_relaxacao)
      df['Amplitude Relaxacao'] = pd.Series(amplitude_sinal_niumag)
      df['Amplitude Relaxacao Fitting'] = pd.Series(amplitude_sinal_fitting)

    if Distribuicao == True:
      df['Tempo Distribuicao'] = pd.Series(tempo_distribuicao)
      df['Distribuicao T2'] = pd.Series(distribuicao_t2)

    if T2_niumag == True:
      df['T2 Geometrico Niumag'] = t2gm_niumag
      df['T2 Medio Niumag'] = t2av_niumag

    if Erro == True:
      df['Fitting Error'] = fitting_erro

    if Amplitudes == True:
      df['Amplitudes'] = amplitudes

    df = df.sort_values(by = 'Amostra')

    return df



def TratamentoDadosRMN(Diretorio_pasta, Arquivo_laboratorio, Dados_niumag,
                       Porosidade_i = False, poro_i = 'Porosidade RMN', T2_log = False, Componentes_t2 = False,
                       Fator_Cimentacao = False, V_artifical = 1.3, V_geral = 2.0,
                       Fracoes_T2Han = False, Fracoes_T2Ge = False, Localizacao = False,
                       Parametros_lab = ['Permeabilidade Gas', 'Porosidade Gas', 'Porosidade RMN'],
                       Geometria = False, EPSG = 4326, Conversao = False, N_Conversao = 32724,
                       BVIFFI = False, tempo = 'Arenito', Fator_Formacao = False, Litofacie = False,
                       Amplitude = False, Dados_porosidade_Transverso = False, N_transverso = 128):
    """
    Esta função trata mesclar os dados já processados de RMN (como processado pela função anterior e que tenha informações da distribuição de tamanho de poros)
    com os dados laboratoriais, que contenham dados de porosidade a gás e de RMN, permeabilidade a gas, litofácies das amostras.

    Args:
        Diretorio_pasta (str): O caminho do diretório onde está o arquivo excel exportado pelo software Niumag.
        Arquivo_laboratorio (str): Nome do arquivo contendo os dados do laboratório em excel.
        Dados_niumag (pandas.DataFrame): DataFrame com as informações selecionadas da distribuição de tamanho de poros.
        Porosidade_i (bool): Caso o usuário queira a transformação do sinal de RMN em porosidade RMN.
        poro_i (str): Nome da coluna com a porosidade que o usuário deseja normalizar o sinal de amplitude da RMN.
        T2_log (bool): Caso o usuário queira calcular o T2_lm proposto por Kenyon et al (1988). OBS: Não está pronto.
        Componentes_t2 (bool): Caso o usuário tenha as componentes que ajustam a curva de relaxação T2.
        Fator_Cimentacao (bool): Caso o usuário queira obter o fator de cimentação.
        V_artifical (int): Fator de cimentação das amostras artificiais.
        V_geral (int): Fator de cimentação das amostras gerais.
        Fracoes_T2Han (bool): Caso o usuário queira retornar as frações da modelagem proposta por Han et al (2018).
        Fracoes_T2Ge (bool): Caso o usuário queira retornar as frações da modelagem proposta por Ge et al (2017).
        Localizacao (bool): Caso o usuário tenha informações sobre a localização das amostras.
        Parametros_lab (list): Informações dos dados laboratórios que o usuário deseja compor no dataframe final.
        Geometria (bool):  Caso o usuário queira retornar a localização em um formato geométrico.
        EPSG (int): Sistema de coordenadas da European Petroleum Survey Group que está a geometria das amostras. Obs: O formato padrão é o WGS84.
        Conversao (bool): Caso o usuário queira converter os o sistema de coordenadas dos dados lab em outro.
        N_Conversao (int): Sistema de coordenadas da European Petroleum Survey Group que o usuário deseja converter a geometrias das amostras. Obs: O formato de conversão padrão é o WTF24M.
        BVIFFI (bool): Caso o usuário queira retornar as frações da modelagem proposta por Coates et al (1999).
        tempo (str): Informações do modelo de tempo que o usuário deseja retornar.
        Fator_Formacao (bool): Caso o usuário queira obter o fator de formação.
        Litofacie (bool): Caso o usuário tenha nos dados do laboratório informações sobre as litofácies.
        Amplitude (bool): Caso o usuário queira transformar a lista com os valores da Amplitude do sinal de Relaxação em colunas
        Dados_porosidade_Transverso (bool): Caso o usuário queira transformar a lista com os valores da Distribuição de Tamanho de poros em colunas.
        N_transverso (int): Quantidade de pontos da inversão da curva de relaxação T2.

    Returns:
        pandas.DataFrame: Retorna um DataFrame com dados processados do Excel exportado pelo usuário com os dados do laboratório mesclado com os dados da Niumag.

    Exemplos de Uso:
        Caso o usuário tenha um arquivo .xlsx com dados de laboratório e um pandas.DataFrame com dados de Distribuição de Tamanho de Poros
        essa função retornará os dados mesclados e prontos para regressões ou visualizações no formato pandas.DataFrame.
    """

    dados_niumag = Dados_niumag
    dados_lab = pd.read_excel(Diretorio_pasta + Arquivo_laboratorio)
    dados_lab['Amostra'] = dados_lab['Amostra'].astype(str)
    dados_lab = dados_lab.sort_values(by = 'Amostra')

    tempo_distribuicao = dados_niumag['Tempo Distribuicao']
    distribuicao_t2 = dados_niumag['Distribuicao T2']

    porosi_i = []
    media_ponderada_log = []
    s1h = []
    s2h = []
    s3h = []
    s4h = []
    s1g = []
    s3g = []
    s4g = []
    BVI = []
    FFI = []
    A1 = []
    A2 = []
    A3 = []

    df_niumag_sa = dados_niumag.drop('Amostra', axis = 1)
    df = pd.concat([dados_lab[Parametros_lab], df_niumag_sa, ], axis = 1)

    if Litofacie == True:
      codi_lab = preprocessing.LabelEncoder()
      categoria_lito = codi_lab.fit_transform(dados_lab['Litofacies'])
      onehot = OneHotEncoder()
      ohe = pd.DataFrame(onehot.fit_transform(dados_lab[['Litofacies']]).toarray())
      ohe.columns = onehot.categories_
      df['Litofacies'] = dados_lab['Litofacies']
      df['Categoria Litofacie'] = categoria_lito
      df = pd.concat([df, ohe], axis = 1)
      # Criação de colunas com valor de 0 ou 1 para cada litofácie

    if Localizacao == True:
      df = pd.concat([df, dados_lab[['X_loc', 'Y_loc', 'Z_loc', 'Caliper']]], axis = 1)

      if Geometria == True:
        geometria = [Point(xy) for xy in zip(dados_lab['X_loc'], dados_lab['Y_loc'])]
        gdf = gpd.GeoDataFrame(dados_lab, geometry=geometria)
        gdf.set_crs(epsg=EPSG, inplace=True)

        df['Geometria EPSG: ' + str(EPSG)] = gdf['geometry']

        if Conversao == True:
          gdf_conv = gdf.to_crs(epsg=N_Conversao)
          df['Geometria EPSG: ' + str(N_Conversao)] = gdf_conv['geometry']
      # Criação da coluna com a geometria do Poço


    if Porosidade_i == True:
      for i in np.arange(len(distribuicao_t2)):
          t2_transpose = pd.DataFrame([distribuicao_t2[i]]).T.reset_index().drop('index', axis = 1).T
          scaler = t2_transpose.T
          scaler_sum_phi = float(dados_lab[str(poro_i)][i])/float(scaler.sum())
          phi_i = []
          for j in np.arange(len(scaler)):
              sc = scaler.T
              p = float(sc[j]*scaler_sum_phi)
              phi_i.append(p)
          porosi_i.append(list(phi_i))
    # Criação de uma lista de valores com distribuição de porosidade

    if Porosidade_i == True:
      df['Porosidade_i'] = porosi_i
      # Criação da coluna com a porosidade

    if T2_log == True:
      for i in np.arange(len(porosi_i)):
        phi_i = porosi_i[i]
        tempo_log = np.log(np.array(tempo_distribuicao[i].T.reset_index().drop('index', axis = 1), dtype = float).T[0])
        produto_porosidade_t2_log = pd.DataFrame(phi_i*tempo_log)
        sum_num = np.sum(produto_porosidade_t2_log)
        sum_den = np.sum(phi_i)
        razao_t2 = float(np.exp(sum_num/sum_den))
        media_ponderada_log.append((razao_t2))
    # Calculando o T2_log usado na função do Kenyon et al (1988)

    if Componentes_t2 == True:
      df = pd.concat([df, dados_lab[['A_NMR', 'T2_NMR',
                                     'A1', 'T21',
                                     'A2', 'T22',
                                     'A3', 'T23']]], axis = 1)
    # Unificação dos dados com as componentes T2

    if Fator_Cimentacao == True:
      def DefinirValor(litofacies):
        if litofacies == 'Artificial':
            return V_artifical
        else:
            return V_geral
      df['Fator de Cimentacao'] = dados_lab['Litofacies'].apply(DefinirValor)
    # Incremento do valor do Fator de Cimentação

    if Fator_Formacao == True:
      df['Fator de Formacao'] = 1/dados_lab['Porosidade RMN']**df['Fator de Cimentacao']
    # Incremento do valor do Fator de Formação

    if Fracoes_T2Han == True:
      for i in np.arange(len(porosi_i)):
        phi_i = pd.Series(porosi_i[i])
        porosidade = np.sum(porosi_i[i])
        a1h = phi_i[:74].sum()
        a2h = phi_i[74:84].sum()
        a3h = phi_i[84:92].sum()
        a4h = phi_i[92:].sum()
        phimicroh = a1h/porosidade
        phimesoh  = a2h/porosidade
        phimacroh = a3h/porosidade
        phisuperh = a4h/porosidade

        if phimicroh <= 0.0001:
          phimicroh = 0.0001
        if phimesoh <= 0.0001:
          phimesoh = 0.0001
        if phimacroh <= 0.0001:
          phimacroh = 0.0001
        if phisuperh <= 0.0001:
          phisuperh = 0.0001

        s1h.append(phimicroh)
        s2h.append(phimesoh)
        s3h.append(phimacroh)
        s4h.append(phisuperh)

      df['S1Han'] = s1h
      df['S2Han'] = s2h
      df['S3Han'] = s3h
      df['S4Han'] = s4h
      # Cálculo das frações da modelagen Han et al (2018)

    if Fracoes_T2Ge == True:
      for i in np.arange(len(porosi_i)):
        phi_i = pd.Series(porosi_i[i])
        phimicrog = phi_i[:75].sum()
        phimesog = phi_i[75:84].sum()
        phimacrog = phi_i[84:91].sum()
        phisuperg = phi_i[91:].sum()


        if phimicrog <= 0.0001:
                phimicrog = 0.0001
        if phimacrog <= 0.0001:
                phimacrog = 0.0001
        if phisuperg <= 0.0001:
                phisuperg = 0.0001

        s1g.append(phimicrog)
        s3g.append(phimacrog)
        s4g.append(phisuperg)

      df['S1Ge'] = s1g
      df['S3Ge'] = s3g
      df['S4Ge'] = s4g
      # Cálculo das frações da modelagen Ge et al (2016)

    if BVIFFI == True:
      for i in np.arange(len(porosi_i)):

        if tempo == 'Arenito':
          phi_i = pd.Series(porosi_i[i])
          b = phi_i[:76].sum()
          f = phi_i[76:].sum()


          if b <= 0.0001:
                b = 0.0001
          if f <= 0.0001:
                f = 0.0001

          BVI.append(b)
          FFI.append(f)

        if tempo == 'Carbonato':
          phi_i = pd.Series(porosi_i[i])
          b = phi_i[:86].sum()
          f = phi_i[86:].sum()


          if b <= 0.0001:
                b = 0.0001
          if f <= 0.0001:
                f = 0.0001

          BVI.append(b)
          FFI.append(f)

      df['BVI'] = BVI
      df['FFI'] = FFI
      # Cálculo do BVI e FFI para modelagem Coates et al (1999)

    if Dados_porosidade_Transverso == True:
      dataframe_porosidade = df['Porosidade_i']
      array_amostras = df['Amostra']
      dados_T = pd.DataFrame([[0 for col in range(N_transverso)] for row in range(len(array_amostras))])
      colunas = []
      for i in range(len(array_amostras)):
        for j in np.arange(N_transverso):
          por = dataframe_porosidade[i][j]
          tempo_distribuido = np.array(df_niumag['Tempo Distribuicao'][i].reset_index().drop('index', axis = 1).T[j])
          string = 'T2 ' + str(tempo_distribuido)[1:-1]
          colunas.append(string)
          dados_T[j][i] = por
      dados_T.columns = colunas[0:N_transverso]
      df = pd.concat([df, dados_T], axis = 1)
      # Transformação da lista em colunas

    if Amplitude == True:
      for i in np.arange(len(dados_niumag['Amplitudes'])):
        lista = []
        for j in np.arange(len(dados_niumag['Amplitudes'][i])):
          a = str(list(dados_niumag['Amplitudes'][i].reset_index().drop('index', axis = 1).T[j]))[1:-1]
          nome = "T2 " + a

          if nome == 'T2 0.0':
              lista.append(0)
          elif nome == 'T2 10000.0':
              lista.append(df['T2 10000'][0])
          else:
              lista.append(df[nome][0])

        if lista[0] == 0:
            lista[0] = 0.000001
        if lista[1] == 0:
            lista[1] = 0.000001
        if lista[2] == 0:
            lista[2] = 0.000001

        A1.append(lista[0])
        A2.append(lista[1])
        A3.append(lista[2])

      df["Amp1"] = A1
      df['Amp2'] = A2
      df['Amp3'] = A3

    df = df.sort_values(by = 'Amostra')

    return df


def ProcessamentoDadosSDR (Dados):

  """
    Seleciona do DataFrame informado apenas os parâmetros necessário para realizar a regressção proposta por Kenyon et al (1988).

    Args:
        Dados (pandas.DataFrame): DataFrame com os dados da amostra, litofácies, o valor de T2_lm, porosidade RMN e Gás, e permeabilidade Gas.

    Returns:
        pandas.DataFrame: Retorna um Dataframe menor com as informações essenciais para a regressão dos dados de permeabilidade proposta por Kenyon et al (1988).

    """
    dados = pd.DataFrame({
        'Amostra': Dados['Amostra'],
        'Litofacies': Dados['Litofacies'],
        'T2': Dados['T2 Ponderado Log'],
        'Porosidade RMN': Dados['Porosidade RMN'],
        'Porosidade Gas': Dados['Porosidade Gas'],
        'Permeabilidade Gas': Dados['Permeabilidade Gas']
        })

    return dados

def ProcessamentoDadosCoates (Dados):

  """
    Seleciona do DataFrame informado apenas os parâmetros necessário para realizar a regressção proposta por Coates et al (1999).

    Args:
        Dados (pandas.DataFrame): DataFrame com os dados da amostra, litofácies, o valor de BVI, FFI, porosidade RMN e Gás, e permeabilidade Gas.

    Returns:
        pandas.DataFrame: Retorna um Dataframe menor com as informações essenciais para a regressão dos dados de permeabilidade proposta por Coates et al (1999).

    """
    dados = pd.DataFrame({
        'Amostra': Dados['Amostra'],
        'Litofacies': Dados['Litofacies'],
        'BVI': Dados['BVI'],
        'FFI': Dados['FFI'],
        'Porosidade RMN': Dados['Porosidade RMN'],
        'Porosidade Gas': Dados['Porosidade Gas'],
        'Permeabilidade Gas': Dados['Permeabilidade Gas']
        })
    return dados

def ProcessamentoDadosHan (Dados):

  """
    Seleciona do DataFrame informado apenas os parâmetros necessário para realizar a regressção proposta por Han et al (2018).

    Args:
        Dados (pandas.DataFrame): DataFrame com os dados da amostra, litofácies, o valor das frações da curva no tempo de corte S1, S2, S3 e S$, porosidade RMN e Gás, e permeabilidade Gas.

    Returns:
        pandas.DataFrame: Retorna um Dataframe menor com as informações essenciais para a regressão dos dados de permeabilidade proposta por Han et al (2018).

    """
    dados = pd.DataFrame({'Amostra': Dados['Amostra'],
                          'Litofacies': Dados['Litofacies'],
                          'Permeabilidade Gas': Dados['Permeabilidade Gas'],
                          'Porosidade Gas': Dados['Porosidade Gas'],
                          'Porosidade RMN': Dados['Porosidade RMN'],
                          'S1Han': Dados['S1Han'],
                          'S2Han': Dados['S2Han'],
                          'S3Han': Dados['S3Han'],
                          'S4Han': Dados['S4Han']
                          }).replace(0, np.nan).dropna().reset_index().drop('index', axis = 1)

    return dados

def ProcessamentoDadosGe (Dados):

  """
    Seleciona do DataFrame informado apenas os parâmetros necessário para realizar a regressção proposta por Ge et al (2017).

    Args:
        Dados (pandas.DataFrame): DataFrame com os dados da amostra, litofácies, o valor das frações da curva no tempo de corte S1, S3 e S$, porosidade RMN e Gás, e permeabilidade Gas.

    Returns:
        pandas.DataFrame: Retorna um Dataframe menor com as informações essenciais para a regressão dos dados de permeabilidade proposta por Ge et al (2017).

    """
    dados = pd.DataFrame({'Amostra': Dados['Amostra'],
                          'Litofacies': Dados['Litofacies'],
                          'Permeabilidade Gas': Dados['Permeabilidade Gas'],
                          'Porosidade Gas': Dados['Porosidade Gas'],
                          'Porosidade RMN': Dados['Porosidade RMN'],
                          'S1Ge': Dados['S1Ge'],
                          'S3Ge': Dados['S3Ge'],
                          'S4Ge': Dados['S4Ge']
                          }).replace(0, np.nan).dropna().reset_index().drop('index', axis = 1)

    return dados


def ProcessamentoDistribuicaoTreinoTeste (Dados_Treino, Dados_Teste,
                                          Valores = ['T2 0.01',  'T2 0.011',  'T2 0.012',  'T2 0.014',  'T2 0.015',  'T2 0.017',  'T2 0.019',  'T2 0.021',  'T2 0.024',
                                          'T2 0.027',  'T2 0.03',  'T2 0.033',  'T2 0.037',  'T2 0.041',  'T2 0.046',  'T2 0.051',  'T2 0.057',  'T2 0.064',
                                          'T2 0.071',  'T2 0.079',  'T2 0.088',  'T2 0.098',  'T2 0.109',  'T2 0.122',  'T2 0.136',  'T2 0.152',  'T2 0.169',
                                          'T2 0.189',  'T2 0.21',  'T2 0.234',  'T2 0.261',  'T2 0.291',  'T2 0.325',  'T2 0.362',  'T2 0.404',  'T2 0.45',
                                          'T2 0.502',  'T2 0.56',  'T2 0.624',  'T2 0.696',  'T2 0.776',  'T2 0.865',  'T2 0.964',  'T2 1.075',  'T2 1.199',
                                          'T2 1.337',  'T2 1.49',  'T2 1.661',  'T2 1.852',  'T2 2.065',  'T2 2.303',  'T2 2.567',  'T2 2.862',  'T2 3.191',
                                          'T2 3.558',  'T2 3.967',  'T2 4.423',  'T2 4.931',  'T2 5.497',  'T2 6.129',  'T2 6.834',  'T2 7.619',  'T2 8.494',
                                          'T2 9.471',  'T2 10.559',  'T2 11.772',  'T2 13.125',  'T2 14.634',  'T2 16.315',  'T2 18.19',  'T2 20.281',  'T2 22.612',
                                          'T2 25.21',  'T2 28.107',  'T2 31.337',  'T2 34.939',  'T2 38.954',  'T2 43.431',  'T2 48.422',  'T2 53.986',  'T2 60.19',
                                          'T2 67.108',  'T2 74.82',  'T2 83.418',  'T2 93.004',  'T2 103.693',  'T2 115.609',  'T2 128.895',  'T2 143.708',  'T2 160.223',
                                          'T2 178.636',  'T2 199.165',  'T2 222.053',  'T2 247.572',  'T2 276.023',  'T2 307.744',  'T2 343.11',  'T2 382.54',  'T2 426.502',
                                          'T2 475.516',  'T2 530.163',  'T2 591.09',  'T2 659.019',  'T2 734.754',  'T2 819.192',  'T2 913.335',  'T2 1018.296',  'T2 1135.32',
                                          'T2 1265.792',  'T2 1411.258',  'T2 1573.441',  'T2 1754.262',  'T2 1955.864',  'T2 2180.633',  'T2 2431.234',  'T2 2710.634',  'T2 3022.143',
                                          'T2 3369.45',  'T2 3756.671',  'T2 4188.391',  'T2 4669.725',  'T2 5206.375',  'T2 5804.697',  'T2 6471.778',  'T2 7215.521',  'T2 8044.736',
                                          'T2 8969.245',  'T2 10000']):
  
    """
    Seleciona do DataFrame informado as colunas da distribuição de Tamanho de Poros.

    Args:
        Dataframe (pandas.DataFrame): DataFrame com os dados da amostra, litofácies, o valor das frações da curva no tempo de corte S1, S3 e S$, porosidade RMN e Gás, e permeabilidade Gas.

    Returns:
        X_treino (pandas.DataFrame): Retorna um Dataframe com as informações essenciais para a regressão dos dados de permeabilidade que utilizem a curva de distribuição de tamanho de poros.
        X_teste (pandas.DataFrame): Retorna um Dataframe com as informações essenciais para a avaliação dos dados de permeabilidade que utilizem a curva de distribuição de tamanho de poros.
        y_treino (numpy.array): Retorna um numpy.array com as informações da permeabilidade de cada amostra para treinamento e regressão.
        y_teste (numpy.array): Retorna um numpy.array com as informações da permeabilidade de cada amostra para avaliação do modelo.

    """

    X_treino = Dados_Treino[Valores]
    y_treino = np.log10(Dados_Treino['Permeabilidade Gas']*1000)
    
    X_teste = Dados_Teste[Valores]
    y_teste = np.log10(Dados_Teste['Permeabilidade Gas']*1000)
    
    return X_treino, y_treino, X_teste, y_teste

def ProcessamentoReservatorio (Dados_com_Previsao, Modelagens = ['SDR']):

  
  """
    Realiza o processamento dos dados com valores de permeabilidade(s) modelada(s) a fim de obter parâmetros petrofísicos de reservatório proposta por Soto et al (2010).

    Args:
        Dados_com_Previsao (pandas.DataFrame): DataFrame com os dados de permeabilidade a gás e porosidade a gás e RMN.
        Modelagens (list): Lista com os nomes das modelagens que necessita de avaliação dos parâmetros do reservatório.

    Returns:
        pandas.DataFrame: Retorna um acréscimo ao DataFrame com os valores dos parâmetros petrofísicos de reservatório.

  """

  k_gas = Dados_com_Previsao['Permeabilidade Gas']
  phi_gas = Dados_com_Previsao['Porosidade Gas']
  phi_rmn = Dados_com_Previsao['Porosidade RMN']

  phi_z_gas = phi_gas/(1-phi_gas)
  phi_z_rmn = phi_rmn/(1-phi_rmn)

  rqi_gas = (np.pi/100)*np.sqrt(k_gas/phi_gas)
  fzi_gas = rqi_gas/phi_z_gas

  polar_arm_gas = phi_z_gas*(np.sqrt(fzi_gas**2 + 1))
  polar_angle_gas = np.arctan(fzi_gas)

  df = pd.DataFrame({'Phi_z_Gas': phi_z_gas,
                     'Phi_z_RMN': phi_z_rmn,
                     'RQI_Gas': rqi_gas,
                     'FZI_Gas': fzi_gas,
                     'Polar_arm_Gas': polar_arm_gas,
                     'Polar_angle_Gas': polar_angle_gas})

  for i in np.arange(len(Modelagens)):
    k_mod = Dados_com_Previsao['Permeabilidade Prevista ' + Modelagens[i]]

    rqi_mod = (np.pi/100)*np.sqrt(k_mod/phi_rmn)
    fzi_mod = rqi_mod/phi_z_rmn

    polar_arm_mod = phi_z_rmn*(np.sqrt(fzi_mod**2 + 1))
    polar_angle_mod = np.arctan(fzi_mod)


    df['RQI_' + Modelagens[i]] = rqi_mod
    df['FZI_' + Modelagens[i]] = fzi_mod
    df['Polar_arm_' + Modelagens[i]] = polar_arm_mod
    df['Polar_angle_' + Modelagens[i]] = polar_angle_mod


  return pd.concat([Dados_com_Previsao, df], axis = 1)


def DadosRidgeLine(Dados, Pasta_salvamento, Nome,
                   Distribuicao = ['T2 0.01',  'T2 0.011',  'T2 0.012',  'T2 0.014',  'T2 0.015',  'T2 0.017',  'T2 0.019',  'T2 0.021',  'T2 0.024',
                   'T2 0.027',  'T2 0.03',  'T2 0.033',  'T2 0.037',  'T2 0.041',  'T2 0.046',  'T2 0.051',  'T2 0.057',  'T2 0.064', 
                   'T2 0.071',  'T2 0.079',  'T2 0.088',  'T2 0.098',  'T2 0.109',  'T2 0.122',  'T2 0.136',  'T2 0.152',  'T2 0.169',
                   'T2 0.189',  'T2 0.21',  'T2 0.234',  'T2 0.261',  'T2 0.291',  'T2 0.325',  'T2 0.362',  'T2 0.404',  'T2 0.45',
                   'T2 0.502',  'T2 0.56',  'T2 0.624',  'T2 0.696',  'T2 0.776',  'T2 0.865',  'T2 0.964',  'T2 1.075',  'T2 1.199',
                   'T2 1.337',  'T2 1.49',  'T2 1.661',  'T2 1.852',  'T2 2.065',  'T2 2.303',  'T2 2.567',  'T2 2.862',  'T2 3.191',  
                   'T2 3.558',  'T2 3.967',  'T2 4.423',  'T2 4.931',  'T2 5.497',  'T2 6.129',  'T2 6.834',  'T2 7.619',  'T2 8.494', 
                   'T2 9.471',  'T2 10.559',  'T2 11.772',  'T2 13.125',  'T2 14.634',  'T2 16.315',  'T2 18.19',  'T2 20.281',  'T2 22.612', 
                   'T2 25.21',  'T2 28.107',  'T2 31.337',  'T2 34.939',  'T2 38.954',  'T2 43.431',  'T2 48.422',  'T2 53.986',  'T2 60.19',
                   'T2 67.108',  'T2 74.82',  'T2 83.418',  'T2 93.004',  'T2 103.693',  'T2 115.609',  'T2 128.895',  'T2 143.708',  'T2 160.223',
                   'T2 178.636',  'T2 199.165',  'T2 222.053',  'T2 247.572',  'T2 276.023',  'T2 307.744',  'T2 343.11',  'T2 382.54',  'T2 426.502',
                   'T2 475.516',  'T2 530.163',  'T2 591.09',  'T2 659.019',  'T2 734.754',  'T2 819.192',  'T2 913.335',  'T2 1018.296',  'T2 1135.32',
                   'T2 1265.792',  'T2 1411.258',  'T2 1573.441',  'T2 1754.262',  'T2 1955.864',  'T2 2180.633',  'T2 2431.234',  'T2 2710.634',  'T2 3022.143',
                   'T2 3369.45',  'T2 3756.671',  'T2 4188.391',  'T2 4669.725',  'T2 5206.375',  'T2 5804.697',  'T2 6471.778',  'T2 7215.521',  'T2 8044.736',
                   'T2 8969.245',  'T2 10000']):

  """
    Cria um DataFrame concatenado em uma única lista para todos os valores de tempo e da distribuição T2 para visualização da distribuição no formato RidgeLine.

    Args:
        Dados (pandas.DataFrame): DataFrame com os dados de distribuição T2 jpa formato em colunas.
        Pasta_Salvamento (str): Diretório onde será salvo o arquivo gerado.
        Nome (str): Nome que deseja salvar esse DataFrame.
        Distribuicao (list): Lista das colunas da distribuição T2

    Returns:
        pandas.DataFrame: Retorna um DataFrame com os valores em lista das distribuições T2 e os tempos respectivos.

  """

    porosidade_i = []
    tempo_distribuicao = []
    for i in np.arange(len(Dados)):
        phi_i = []
        tempo = []
        for j in np.arange(len(Distribuicao)):
            phi_i.append(dados[Distribuicao[j]][i])
            tempo.append(float(Distribuicao[j][3:]))
        porosidade_i.append(phi_i)
        tempo_distribuicao.append(tempo)
    
    Dados['Porosidade i'] = porosidade_i
    Dados['Tempo Distribuicao'] = tempo_distribuicao
    
    lista_tempo = []
    lista_amostra = []
    lista_t2 = []
    lista_litofacie = []
    lista_poço = []
    for i in np.arange(len(dados)):
        for j in np.arange(len(Dados['Tempo Distribuicao'][0])):
            lista_amostra.append(Dados['Amostra'][i])
            lista_tempo.append(Dados['Tempo Distribuicao'][i][j])
            lista_t2.append(Dados['Porosidade i'][i][j])
            lista_litofacie.append(Dados['Categoria Litofacies'][i])
            lista_poço.append(Dados['Poço'][i])

    df = pd.DataFrame({'Amostra': lista_amostra,
                       'Poço': lista_poço,
                       'Tempo': lista_tempo,
                       'T2': lista_t2,
                       'Litofacie': lista_litofacie})
    local_salvamento = Pasta_salvamento + 'Dados_RidgeLine_' + str(Nome) + '.xlsx'
    df.to_excel(local_salvamento, sheet_name='Dados')                          # Salvar dataframe
    
    return df
