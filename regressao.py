# Bibliotecas para leitura e processamentos dos dados
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize, curve_fit
from scipy.stats import chi2

# Bibliotecas para o Projeto RMN
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score


def RegressaoSDR (Dataframe_SDR):

    """
    A regressão dos coeficientes da modelagem SDR proposta por Kenyon et al (1988).

    Args:
        Dataframe_SDR (pandas.DataFrame): Dataframe com os dados necessários para modelagem

    Returns:
        Retorna a regressão realizada (reg_ols_log), os coeficientes da regressão (coeficientes), um dataframe com os dados previstos concatenado
        com o DataFrame informado, e o erro sigma.
    """
    # Regressão via OLS
    t2 = Dataframe_SDR['T2 Ponderado Log']
    phi = Dataframe_SDR['Porosidade RMN']
    permeabilidade = Dataframe_SDR['Permeabilidade Gas']
    dados_calculo = pd.DataFrame({'Log k': np.log(permeabilidade),
                                'Log φ': np.log(phi),
                                'Log T2': np.log(t2)})
    dados_calculo = sm.add_constant(dados_calculo)
    atributos = dados_calculo[['const', 'Log φ', 'Log T2']]
    rotulos = dados_calculo[['Log k']]
    reg_ols_log = sm.OLS(rotulos, atributos, hasconst=True).fit()

    # Obtenção dos coeficientes da Regressão
    coeficientes = pd.DataFrame({
        'Coeficiente': ['a', 'b', 'c', 'R2'],
        'Valor': [np.exp(reg_ols_log.params[0]),
                  reg_ols_log.params[1],
                  reg_ols_log.params[2],
                  reg_ols_log.rsquared]}).set_index('Coeficiente')

    # Cálculo da Previsão com base nos coeficientes obtidos
    a = coeficientes['Valor']['a']
    b = coeficientes['Valor']['b']
    c = coeficientes['Valor']['c']
    k = (a*(phi**b)*(t2**c))
    dados = pd.DataFrame({'Permeabilidade Prevista': k})

    #Erro Sigma
    k_p = np.log10(dados['Permeabilidade Prevista'])
    k_g = np.log10(permeabilidade)
    N = len(k_p)
    soma = np.sum((k_p-k_g)**2)
    raiz = np.sqrt(soma/N)
    sigma = 10**(raiz)

    return reg_ols_log, coeficientes, pd.concat([Dataframe_SDR, dados], axis = 1), sigma

def RegressaoCoates (Dataframe_Coates):

    """
    A regressão dos coeficientes da modelagem Coates proposta por Coates et al (1999).

    Args:
        Dataframe_Coates (pandas.DataFrame): Dataframe com os dados necessários para modelagem.

    Returns:
        Retorna a regressão realizada (reg_ols_log), os coeficientes da regressão (coeficientes), um dataframe com os dados previstos concatenado
        com o DataFrame informado, e o erro sigma.
    """
    # Regressão via OLS
    FFIBVI = Dataframe_Coates['FFI']/Dataframe_Coates['BVI']
    phi = Dataframe_Coates['Porosidade RMN']
    permeabilidade = Dataframe_Coates['Permeabilidade Gas']
    dados_calculo = pd.DataFrame({'Log k': np.log(permeabilidade),
                                'Log φ': np.log(phi),
                                'Log FFI/BVI': np.log(FFIBVI)})
    dados_calculo = sm.add_constant(dados_calculo)
    atributos = dados_calculo[['const', 'Log φ', 'Log FFI/BVI']]
    rotulos = dados_calculo[['Log k']]
    reg_ols_log = sm.OLS(rotulos, atributos, hasconst=True).fit()

    # Obtenção dos coeficientes da Regressão
    coeficientes = pd.DataFrame({
        'Coeficiente': ['a', 'b', 'c', 'R2'],
        'Valor': [np.exp(reg_ols_log.params[0]),
                  reg_ols_log.params[1],
                  reg_ols_log.params[2],

                  reg_ols_log.rsquared]}).set_index('Coeficiente')

    # Cálculo da Previsão com base nos coeficientes obtidos
    a = coeficientes['Valor']['a']
    b = coeficientes['Valor']['b']
    c = coeficientes['Valor']['c']
    k = (a*(phi**b)*(FFIBVI**c))
    dados = pd.DataFrame({'Permeabilidade Prevista': k})

    #Erro Sigma
    k_p = np.log10(dados['Permeabilidade Prevista'])
    k_g = np.log10(permeabilidade)
    N = len(k_p)
    soma = np.sum((k_p-k_g)**2)
    raiz = np.sqrt(soma/N)
    sigma = 10**(raiz)

    return reg_ols_log, coeficientes, pd.concat([Dataframe_Coates, dados], axis = 1), sigma

def RegressaoHan (Dataframe_Han):

    """
    A regressão dos coeficientes da modelagem Coates proposta por Han et al (2018).

    Args:
        Dataframe_Han (pandas.DataFrame): Dataframe com os dados necessários para modelagem.

    Returns:
        Retorna a regressão realizada (reg_novo), os coeficientes da regressão (coeficientes_novo), um dataframe com os dados previstos concatenado
        com o DataFrame informado, e o erro sigma.
    """
    # Regressão via OLS
    dados_calculo_log = pd.DataFrame({
    'Log k': np.log(Dataframe_Han['Permeabilidade Gas']),
    'Log φ': np.log(Dataframe_Han['Porosidade RMN']),
    'S1 log': (-1)*(np.log(Dataframe_Han['S1Han'])),
    'S2 log': (-1)*(np.log(Dataframe_Han['S2Han'])),
    'S3 log': np.log(Dataframe_Han['S3Han']),
    'S4 log': np.log(Dataframe_Han['S4Han'])})
    dados_calculo = sm.add_constant(dados_calculo_log)

    atributos = dados_calculo[['const', 'Log φ', 'S3 log', 'S4 log', 'S1 log', 'S2 log']]
    rotulos = dados_calculo['Log k']
    reg_novo = sm.OLS(rotulos, atributos, hasconst=True, missing = 'drop').fit()

    # Obtenção dos coeficientes da Regressão
    coeficientes_novo = pd.DataFrame({
          'Coeficiente': ['a', 'b', 'c', 'd', 'e', 'f', 'R2'],
          'Valor': [np.exp(reg_novo.params[0]),
                    reg_novo.params[1],
                    reg_novo.params[2],
                    reg_novo.params[3],
                    reg_novo.params[4],
                    reg_novo.params[5],
                    reg_novo.rsquared]
          }).set_index('Coeficiente')

    # Cálculo da Previsão com base nos coeficientes obtidos
    a = coeficientes_novo['Valor']['a']
    b = coeficientes_novo['Valor']['b']
    c = coeficientes_novo['Valor']['c']
    d = coeficientes_novo['Valor']['d']
    e = coeficientes_novo['Valor']['e']
    f = coeficientes_novo['Valor']['f']
    phi = Dataframe_Han['Porosidade RMN']
    s1 = Dataframe_Han['S1Han']
    s2 = Dataframe_Han['S2Han']
    s3 = Dataframe_Han['S3Han']
    s4 = Dataframe_Han['S4Han']
    k = a*(phi**b)*(s3**c)*(s4**d)/((s1**e)*(s2**f))
    dados = pd.DataFrame({'Permeabilidade Prevista': k})

    #Erro Sigma
    k_p = np.log10(dados['Permeabilidade Prevista'])
    k_g = np.log10(Dataframe_Han['Permeabilidade Gas'])
    N = len(k_p)
    soma = np.sum((k_p-k_g)**2)
    raiz = np.sqrt(soma/N)
    sigma = 10**(raiz)



    return reg_novo, coeficientes_novo, pd.concat([Dataframe_Han, dados], axis = 1), sigma


def RegressaoGe (Dataframe_Ge):

    """
    A regressão dos coeficientes da modelagem Coates proposta por Ge et al (2017).

    Args:
        Dataframe_Ge (pandas.DataFrame): Dataframe com os dados necessários para modelagem.

    Returns:
        Retorna a regressão realizada (reg_novo), os coeficientes da regressão (coeficientes_novo), um dataframe com os dados previstos concatenado
        com o DataFrame informado, e o erro sigma.
    """
    # Regressão via OLS
    dados_calculo_log = pd.DataFrame({
    'Log k': np.log(Dataframe_Ge['Permeabilidade Gas']),
    'Log φ': np.log(Dataframe_Ge['Porosidade RMN']),
    'S1 log': (-1)*(np.log(Dataframe_Ge['S1Ge'])),
    'S3Ge': Dataframe_Ge['S3Ge'],
    'S4Ge': Dataframe_Ge['S4Ge']})

    # Função para calcular a soma dos quadrados dos resíduos
    def residuals(params, df):
      ln_a, b, c, d, e = params
      ln_P3c_P4d = np.log(df['S3Ge']**c + df['S4Ge']**d)
      predicted_ln_k = ln_a + b * df['Log φ'] + ln_P3c_P4d - e * df['S1 log']
      return np.sum((df['Log k'] - predicted_ln_k) ** 2)

    # Valores iniciais para os parâmetros
    initial_params = [0, 0, 0, 0, 0]

    # Minimização da função de resíduos
    result = minimize(residuals, initial_params, args=(dados_calculo_log), method='BFGS')

    # Extração dos parâmetros ajustados
    ln_a, b, c, d, e = result.x
    a = np.exp(ln_a)

    # Cálculo de ln(P3^c + P4^d) com os coeficientes ajustados
    dados_calculo_log['Log S3c_S4d'] = np.log(dados_calculo_log['S3Ge']**c + dados_calculo_log['S4Ge']**d)

    # Definindo as variáveis independentes e a variável dependente
    X = dados_calculo_log[['Log φ', 'S1 log', 'Log S3c_S4d']]
    dados_calculo = sm.add_constant(X)  # Adiciona uma constante (intercepto)
    atributos = dados_calculo[['const', 'Log φ', 'Log S3c_S4d', 'S1 log']]
    rotulos = dados_calculo_log['Log k']

    # Ajustando o modelo de regressão
    reg_novo = sm.OLS(rotulos, atributos, hasconst=True, missing = 'drop').fit()

    # Obtenção dos coeficientes da Regressão
    coeficientes_novo = pd.DataFrame({
          'Coeficiente': ['a', 'b', 'c', 'd', 'e', 'R2'],
          'Valor': [np.exp(reg_novo.params[0]),
                    reg_novo.params[1],
                    c,
                    d,
                    reg_novo.params[2],
                    reg_novo.rsquared]
          }).set_index('Coeficiente')

    # Cálculo da Previsão com base nos coeficientes obtidos
    a = coeficientes_novo['Valor']['a']
    b = coeficientes_novo['Valor']['b']
    e = coeficientes_novo['Valor']['e']
    phi = Dataframe_Ge['Porosidade RMN']
    s1Ge = Dataframe_Ge['S1Ge']
    s3Ge = Dataframe_Ge['S3Ge']
    s4Ge = Dataframe_Ge['S4Ge']
    k = a*(phi**b)*((s3Ge**c)+(s4Ge**d)/(s1Ge**e))
    dados = pd.DataFrame({'Permeabilidade Prevista': k})


    #Erro Sigma
    k_p = np.log10(dados['Permeabilidade Prevista'])
    k_g = np.log10(Dataframe_Ge['Permeabilidade Gas'])
    N = len(k_p)
    soma = np.sum((k_p-k_g)**2)
    raiz = np.sqrt(soma/N)
    sigma = 10**(raiz)



    return reg_novo, coeficientes_novo, pd.concat([Dataframe_Ge, dados], axis = 1), sigma


def RegressaoRios(dados_treino, dados_teste):
    

    """
    A regressão dos coeficientes da modelagem Coates proposta por Rios et al (2011).

    Args:
        dados_treino (pandas.DataFrame): Dataframe com os dados necessários para o treinamento do modelo.
        dados_teste (pandas.DataFrame): Dataframe com os dados necessários avaliação do modelo.
    Returns:
        Retorna a regressão realizada (reg_novo), os coeficientes da regressão (coeficientes_novo), um dataframe com os dados previstos concatenado
        com o DataFrame informado, e o erro sigma.
    """
      
    X_treino = dados_treino[['T2 0.01',  'T2 0.011',  'T2 0.012',  'T2 0.014',  'T2 0.015',  'T2 0.017',  'T2 0.019',  'T2 0.021',  'T2 0.024',
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
           'T2 8969.245',  'T2 10000']]
    y_treino = np.log10(dados_treino['Permeabilidade Gas']*1000)

    X_teste = dados_teste[['T2 0.01',  'T2 0.011',  'T2 0.012',  'T2 0.014',  'T2 0.015',  'T2 0.017',  'T2 0.019',  'T2 0.021',  'T2 0.024',
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
            'T2 8969.245',  'T2 10000']]
    y_teste = np.log10(dados_teste['Permeabilidade Gas']*1000)

    pls6 = PLSRegression(n_components=6)
    pls6.fit(X_treino, y_treino)
  
    y_pred_treino = pls6.predict(X_treino)
    y_pred_teste = pls6.predict(X_teste)



    dados_treino['Permeabilidade Prevista Rios'] = (10**y_pred_treino)/1000
    dados_teste['Permeabilidade Prevista Rios'] = (10**y_pred_teste)/1000

    return dados_treino, dados_teste

def RegressaoFZI(dados, modelos):
    
    """
    A regressão FZI.

    Args:
        dados (pandas.DataFrame): Dataframe com os dados necessários para modelagem.
        modelos (list): Lista com os modelos utilizados para obter o FZI.
    Returns:
        Retorna a regressão FZI para cada litofácie.
    """

  lito = dados['Litofacies'].unique()
  coef = []

  for i in np.arange(len(lito)):
    for j in np.arange(len(modelos)):

      df_dados = dados.loc[dados['Litofacies'] == dados['Litofacies'].unique()[i]].reset_index().drop('index', axis = 1)
      rqi = df_dados['RQI_' + modelos[j]]

      if modelos[j] == "Gas":
        phi = df_dados['Phi_z_Gas']

      else:
        phi = df_dados['Phi_z_RMN']

      dados_calculo = pd.DataFrame({'Phi': phi,
                                    'RQI': rqi})

      dados_calculo['const'] = 1

      dados_calculo = sm.add_constant(dados_calculo)
      atributos = dados_calculo[['const', 'Phi']]
      rotulos = dados_calculo[['RQI']]
      reg_ols_log = sm.OLS(rotulos, atributos, hasconst=True).fit()
      coef.append([dados['Litofacies'].unique()[i] + '_' + modelos[j], reg_ols_log.params[0], reg_ols_log.params[1], reg_ols_log.params[0]+reg_ols_log.params[1]])

  c = pd.DataFrame(coef).rename(columns={0: 'Litofacies', 1:'b', 2:'a', 3:'FZI'})
  return c



def RegressaoComponentesT2 (Dados, n = 0, P0 = (1, 0.1), Params_Init = [0.8, 0.001, 0.1, 0.01, 0.1, 0.1]):
    """
    A regressão da curva de relaxação para obter as componentes T2 de uma única .

    Args:
        Dados (pandas.DataFrame): Dataframe com os dados necessários para modelagem.
        n (int): Indice da amostra que terá seus componentes avaliados.
        P0 (tuple): Tupla com oa parâmetros do coeficiente T2_nmr.
        Params_Init (list): Lista de parâmetros iniciais de cada componente T2 OBS: Caso apareça qualquer mensagem de erro ou 
        'O ajuste do modelo não é adequado. Considere revisar os parâmetros iniciais.' mudar esses valores até que a única saida seja
        'O ajuste do modelo é adequado.'
    Returns:
        Retorna um DataFrame com todos os coeficientes T2 e o erro R^2.
    """

  def exponential_decay(t, a, b):
      return a * np.exp(-b * t)

  # Função do modelo exponencial multi-termo
  def multi_exponential_decay(t, params):
      a, b, c, d, g, h = params
      return a * np.exp(-b * t) + c * np.exp(-d * t) + g * np.exp(-h * t)

  # Função de erro (MSE)
  def mse(params, t, y):
      y_pred = multi_exponential_decay(t, params)
      return np.mean((y - y_pred)**2)


  time = np.array(Dados['Tempo Relaxacao'][n])  # Coloque seus valores de tempo aqui
  A_t = np.array(Dados['Amplitude Relaxacao'][n])  # Coloque seus valores de A(t) aqui

  # Realizar o ajuste usando curve_fit
  p0 = P0  # Valores iniciais para a e b
  params, cov = curve_fit(exponential_decay, time, A_t, p0=p0)

  # Parâmetros ajustados
  anmr_fit, bnmr_fit = params

  # Chute inicial para os parâmetros (a, b, c, d, g, h)
  params_init = Params_Init

  # Minimização do erro usando minimize (Método dos mínimos quadrados)
  result = minimize(mse, params_init, args=(time, A_t))

  # Parâmetros ajustados
  a_fit, b_fit, c_fit, d_fit, g_fit, h_fit = result.x

  # Função para calcular as frequências esperadas
  def expected_frequencies(params, t):
      y_pred = multi_exponential_decay(t, params)
      return y_pred

  # Frequências esperadas
  expected_values = expected_frequencies(params_init, time)

  # Cálculo do qui-quadrado
  chi_square_statistic = np.sum((A_t - expected_values)**2 / expected_values)

  # Número de graus de liberdade
  degrees_of_freedom = len(A_t) - len(params_init)

  # Valor crítico para alpha = 0.05 (95% de confiança) e graus de liberdade
  critical_value = chi2.ppf(0.95, degrees_of_freedom)

  # Comparação com o valor crítico
  if chi_square_statistic <= critical_value:
      print('O ajuste do modelo é adequado.')
  else:
      print('O ajuste do modelo não é adequado. Considere revisar os parâmetros iniciais.')

  coef = pd.DataFrame({'Amostra': Dados['Amostra'][n],
                       'Amplitude Relaxacao': [A_t],
                       'Tempo Relaxacao': [time],
                       'A_NMR': [anmr_fit],
                       'T2_NMR': [1/bnmr_fit]})
  coef['A1'] = [a_fit]
  coef['T21'] = [1/b_fit]
  coef['A2'] = [c_fit]
  coef['T22'] = [1/d_fit]
  coef['A3'] = [g_fit]
  coef['T23'] = [1/h_fit]

  ft = coef['A_NMR'][0] * np.exp((-1/coef['T2_NMR'][0]) * time)
  f1 = coef['A1'][0] * np.exp((-1/coef['T21'][0]) * time)
  f2 = coef['A2'][0] * np.exp((-1/coef['T22'][0]) * time)
  f3 = coef['A3'][0] * np.exp((-1/coef['T23'][0]) * time)

  r2_ft = r2_score(A_t, ft)
  r2_fc = r2_score(A_t, f1+f2+f3)

  coef['R2_FT'] = r2_ft
  coef['R2_FC'] = r2_fc


  return coef
