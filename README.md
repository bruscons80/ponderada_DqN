# Deep Q-Network (DQN) aplicado ao Lunar Lander

## 1. Compreensão Documentada do Ambiente

### (a) Variáveis de Estado do Lunar Lander

O ambiente retorna um vetor de observação com 8 dimensões contínuas a cada passo. Inspecionando diretamente `env.observation_space` no notebook (célula de exploração do ambiente), o espaço declarado é:

```
Box([-2.5, -2.5, -10.0, -10.0, -6.2831855, -10.0, -0.0, -0.0],
    [ 2.5,  2.5,  10.0,  10.0,  6.2831855,  10.0,  1.0,  1.0], (8,), float32)
```

| Índice | Variável | Significado Físico | Limite Técnico (Box) | Faixa Típica em Voo Normal |
|--------|----------|-------------------|----------------------|----------------------------|
| 0 | Posição x | Deslocamento horizontal do módulo em relação ao centro da plataforma de pouso | [-2.5, 2.5] | [-1.0, 1.0] |
| 1 | Posição y | Altura do módulo acima do solo (0 ≈ nível do solo) | [-2.5, 2.5] | [0.0, 1.5] |
| 2 | Velocidade x | Componente horizontal da velocidade do módulo | [-10.0, 10.0] | [-1.0, 1.0] |
| 3 | Velocidade y | Componente vertical da velocidade (negativo indica descida) | [-10.0, 10.0] | [-1.5, 0.5] |
| 4 | Ângulo θ | Inclinação do módulo em relação ao eixo vertical | [-2π, 2π] ≈ [-6.28, 6.28] | [-0.5, 0.5] |
| 5 | Velocidade angular | Taxa de variação do ângulo θ por unidade de tempo | [-10.0, 10.0] | [-1.0, 1.0] |
| 6 | Contato perna esquerda | Indicador booleano: 1 se a perna esquerda toca o solo | {0, 1} | {0, 1} |
| 7 | Contato perna direita | Indicador booleano: 1 se a perna direita toca o solo | {0, 1} | {0, 1} |

As 6 primeiras variáveis são contínuas e observáveis diretamente pela física do simulador Box2D. As 2 últimas são binárias e funcionam como sensores de toque. O limite técnico declarado pelo `Box` é amplo (permite estados fisicamente extremos, como ângulos de várias voltas), mas na prática um episódio que não termina em crash imediato permanece dentro da faixa típica indicada na última coluna. No estado inicial amostrado com `SEED=42`, por exemplo, o módulo nasce em `pos_y ≈ 1.42` (alto, no topo do céu) com velocidades pequenas (`vel_x ≈ 0.23`, `vel_y ≈ 0.32`) e ângulo próximo de zero, sem contato de pernas — consistente com a faixa típica.

### (b) Ações Disponíveis

O espaço de ações é discreto com 4 opções mutuamente exclusivas:

| Ação | Propulsor Ativado | Efeito Físico no Módulo |
|------|------------------|------------------------|
| 0 | Nenhum | O módulo está sujeito apenas à gravidade e inércia |
| 1 | Propulsor lateral esquerdo | Empurra o módulo para a direita; gera torque no sentido anti-horário |
| 2 | Propulsor principal (inferior) | Empurra o módulo para cima, desacelerando a queda |
| 3 | Propulsor lateral direito | Empurra o módulo para a esquerda; gera torque no sentido horário |

A ação 2 (propulsor principal) é a mais poderosa e a que mais consome "combustível" em termos de penalização. As ações 1 e 3 controlam a orientação do módulo. Uma política ótima alterna entre essas ações para manter a trajetória alinhada com a plataforma enquanto desacelera.

### (c) Estrutura da Função de Recompensa

A função de recompensa combina sinais de aproximação, estabilidade e consumo de propulsão:

**Recompensas positivas:**
- Aproximação da plataforma de pouso: recompensa proporcional à redução da distância horizontal e vertical e à redução das velocidades linear e angular
- Contato das pernas com o solo: +10 pontos por perna que toca o solo
- Pouso suave e bem-sucedido (módulo para completamente): +100 pontos de bônus

**Penalizações:**
- Afastamento da plataforma ou aumento de velocidade: recompensa negativa proporcional ao aumento da distância ou velocidade
- Acionamento do propulsor principal (ação 2): -0.3 por passo em que é ativado
- Acionamento dos propulsores laterais (ações 1 e 3): -0.03 por passo em que são ativados
- Crash (colisão do corpo principal com o solo ou saída dos limites): -100 pontos de penalidade

A recompensa acumulada de um episódio bem-sucedido típico varia entre +200 e +300 pontos. Episódios com crash acumulam recompensas fortemente negativas.

### (d) Condições de Término de um Episódio

Um episódio termina quando qualquer uma das seguintes condições é atendida:

| Condição | Tipo | Critério |
|----------|------|---------|
| Crash | `terminated=True` | O corpo principal do módulo (não as pernas) toca o solo ou o módulo colide violentamente |
| Saída dos limites | `terminated=True` | O módulo sai dos limites horizontais do ambiente (x < -1 ou x > 1 de forma aproximada) |
| Pouso com sucesso | `terminated=True` | O módulo para completamente na plataforma com ambas as pernas em contato e velocidades próximas de zero |
| Timeout | `truncated=True` | O episódio atinge 1000 passos de tempo sem terminar naturalmente |

No código, `done = terminated or truncated`. O sinal de erro de Bellman usa apenas `terminated` (não `truncated`) para calcular o desconto correto: se o episódio terminou por truncamento de tempo, o estado seguinte ainda tem valor futuro; se terminou por razão física (crash ou pouso), o valor futuro é zero.

### (e) Critério de Pouso Bem-Sucedido e Avaliação de Aprendizado

Um pouso é considerado bem-sucedido quando o módulo para completamente sobre a plataforma de pouso com as duas pernas em contato com o solo, velocidades linear e angular próximas de zero, e o corpo principal sem tocar o terreno.

O critério de desempenho padrão adotado pelo Gymnasium e pela literatura é:

> **Recompensa média acumulada ≥ 200 pontos em 100 episódios consecutivos**

Valores abaixo de 0 indicam que o agente está crashando consistentemente. Valores entre 0 e 100 indicam aprendizado parcial (o agente tenta pousar mas sem controle suficiente). Valores acima de 200 indicam que o agente aprendeu uma política robusta de pouso.

### (f) Por Que Métodos Tabulares Não São Viáveis

O espaço de estados do Lunar Lander é **contínuo e de alta dimensão**: 6 das 8 variáveis de estado assumem valores reais em intervalos contínuos (posições, velocidades, ângulo, velocidade angular). Uma tabela Q exigiria enumerar todos os estados possíveis, o que é inviável para espaços contínuos.

Mesmo com discretização grosseira — por exemplo, 20 bins por variável contínua — a tabela teria 20^6 × 2^2 = 256 milhões de entradas para cada uma das 4 ações, totalizando mais de 1 bilhão de valores Q. Isso seria:
1. **Inviável em memória**: gigabytes de memória para armazenar a tabela
2. **Impossível de explorar**: o agente precisaria visitar cada combinação de estado-ação inúmeras vezes para estimativas confiáveis
3. **Sem generalização**: estados visitados não informam nada sobre estados próximos não visitados

A aproximação de função via rede neural resolve esses problemas: representa Q(s,a) de forma compacta com poucos milhares de parâmetros e generaliza implicitamente para estados não visitados por meio dos pesos compartilhados aprendidos.

---

## 2. Implementação do Agente DQN

### (a) Arquitetura da Rede Neural

A rede neural implementada é um Perceptron Multicamada (MLP) com a seguinte estrutura:

```
Entrada:        8 neurônios   (dimensão do vetor de estado do LunarLander)
Camada oculta 1: 256 neurônios + ReLU
Camada oculta 2: 256 neurônios + ReLU
Saída:          4 neurônios   (um Q-valor por ação discreta)
```

**Justificativas de cada decisão arquitetural:**

- **Entrada = 8**: coerente com o espaço de observação do LunarLander, que retorna um vetor de 8 dimensões conforme descrito na seção 1(a)
- **Saída = 4**: coerente com o espaço de ações discreto de 4 opções; cada neurônio de saída representa Q(s, aᵢ) para a ação i
- **256 neurônios por camada oculta**: capacidade suficiente para aproximar a função Q não-linear do Lunar Lander. A rede resultante tem **69.124 parâmetros treináveis** (verificado no notebook: 8×256 + 256 + 256×256 + 256 + 256×4 + 4 = 69.124), um tamanho que treina em poucos milissegundos por batch mesmo em CPU. Camadas menores (64, 128) tendem a ter capacidade insuficiente para esse espaço de estados contínuo; camadas muito maiores (512+) aumentam o custo computacional por passo sem necessidade, já que o problema tem apenas 8 entradas
- **ReLU como função de ativação**: evita o problema de gradiente evanescente (vanishing gradient) que ocorre com sigmoid e tanh em redes mais profundas; computacionalmente eficiente; a derivada é 0 ou 1, facilitando o fluxo de gradiente
- **Saída linear (sem ativação na última camada)**: os Q-valores podem ser qualquer número real (positivos ou negativos), então não faz sentido restringi-los com sigmoid ou tanh
- **Sem BatchNorm**: a natureza não-estacionária do treinamento DQN (a distribuição dos dados muda conforme a política muda) torna a normalização em batch problemática
- **Sem Dropout**: não se aplica ao controle de agente; a regularização necessária vem da diversidade das experiências no replay buffer

**Seed fixa (SEED = 42):**

A seed é fixada em todas as fontes de aleatoriedade do experimento:
- `random.seed(SEED)`: controla a exploração epsilon-greedy
- `np.random.seed(SEED)`: controla a amostragem do replay buffer
- `torch.manual_seed(SEED)`: controla a inicialização dos pesos da rede
- `env.reset(seed=SEED + episode)`: controla a geração de episódios do ambiente

Sem fixar a seed, diferentes execuções do mesmo código produzem curvas de aprendizado distintas, impossibilitando comparações válidas entre configurações. Com a seed fixa, experimentos com e sem target network partem das mesmas condições iniciais e atravessam os mesmos episódios de treinamento, tornando as diferenças observadas atribuíveis exclusivamente ao componente alterado.

### (b) Loop de Treinamento e Loss Function

O loop de treinamento implementa o algoritmo DQN original de Mnih et al. (2015). A loss function é o erro quadrático médio do erro de Bellman (TD error):

```
L(θ) = E[ ( r + γ · max_{a'} Q_target(s', a') · (1 - done) - Q_online(s, a; θ) )² ]
```

Onde:
- `r` é a recompensa imediata
- `γ = 0.99` é o fator de desconto
- `Q_target(s', a')` são os Q-valores calculados pela target network (pesos congelados)
- `Q_online(s, a; θ)` é o Q-valor da ação tomada, calculado pela rede online
- `done` é 1 se o episódio terminou por razão física (terminated), 0 caso contrário

O alvo é calculado com `torch.no_grad()` para não propagar gradientes através dele. O otimizador Adam atualiza os pesos θ minimizando L(θ) por descida de gradiente estocástico. Gradient clipping (norma máxima = 1.0) é aplicado para evitar explosão de gradientes.

### (c) Experience Replay e Target Network

**Experience Replay (Replay Buffer):**

Sem experience replay, o agente aprende de transições coletadas sequencialmente durante o episódio atual. Essas transições são altamente correlacionadas no tempo: estados consecutivos são quase idênticos, ações são similares, e a distribuição dos dados muda radicalmente conforme a política muda. Redes neurais treinadas com dados temporalmente correlacionados tendem a:
1. Sobreajustar para o comportamento recente, esquecendo o que foi aprendido antes (catastrophic forgetting)
2. Apresentar alta variância nas atualizações de gradiente, dificultando a convergência

O replay buffer armazena as últimas 50.000 transições `(s, a, r, s', done)` e amostra mini-batches aleatórios de 64 transições a cada passo de treinamento. Isso quebra a correlação temporal, diversifica os dados de treinamento e reutiliza cada experiência múltiplas vezes, aumentando a eficiência amostral.

**O que acontece sem experience replay**: o agente treina em dados sequencialmente correlacionados. As atualizações de gradiente oscilam fortemente, o aprendizado é instável, e o agente raramente converge para uma política competente dentro de um número razoável de episódios.

**Target Network (Rede Alvo):**

Sem uma target network separada, o alvo do erro de Bellman é calculado usando os mesmos pesos θ que estão sendo atualizados:

```
target = r + γ · max_{a'} Q(s', a'; θ)
```

Isso cria um problema de "alvo em movimento" (moving target): cada atualização de θ muda simultaneamente o preditor Q(s, a; θ) e o alvo Q(s', a'; θ). O sistema de aprendizado persegue um alvo que se move a cada passo, o que pode criar ciclos de feedback positivo e divergência dos Q-valores.

A target network é uma cópia dos pesos da rede online que é atualizada de forma infrequente (a cada 10 episódios neste trabalho). Durante os episódios entre atualizações, o alvo permanece fixo, tornando o problema de aprendizado mais próximo de uma regressão supervisionada estacionária.

**O que acontece sem target network**: os Q-valores tendem a oscilar mais durante o treinamento e o progresso da média móvel é menos previsível, podendo desacelerar por longos trechos e depois acelerar abruptamente. O experimento 2 (seção 3) mostra esse padrão de forma concreta: durante quase todo o treinamento a configuração sem target network ficou atrás da configuração completa, com uma alta tardia e abrupta perto do final — exatamente o tipo de instabilidade que a target network busca suavizar, mesmo que o resultado final agregado tenha sido parecido nessa execução específica.

---

## 3. Análise Comparativa de Configurações

### Experimentos Realizados

Foram conduzidos três experimentos, todos com `SEED = 42` fixo e `num_episodes = 700`:

| Experimento | Experience Replay | Target Network | Descrição |
|-------------|------------------|---------------|-----------|
| DQN Completo | Sim (buffer=50k, batch=64) | Sim (update a cada 10 eps) | Algoritmo DQN original completo |
| Sem Target Network | Sim (buffer=50k, batch=64) | Não (online net usada como alvo, atualizada a cada passo) | Ablação: remove target network |
| Sem Experience Replay | Não (treina em cada transição individualmente) | Sim (update a cada 10 eps) | Ablação: remove replay buffer |

Hiperparâmetros comuns: γ=0.99, lr=5×10⁻⁴, ε₀=1.0, ε_min=0.01, ε_decay=0.995.

### Curvas de Aprendizado

As curvas de aprendizado completas (recompensa bruta por episódio e média móvel de 100 episódios) estão no notebook e salvas em `comparacao_configuracoes.png`. Resumo numérico extraído da execução real (SEED=42, 700 episódios máximos):

| Configuração | Episódio de Solução | Avg(100) Final | Recompensa Máxima | Resolvido? |
|--------------|---------------------|-----------------|--------------------|-----------|
| DQN Completo | 616 | 200.8 | 315.1 | Sim |
| Sem Target Network | 571 | 200.2 | 314.3 | Sim |
| Sem Experience Replay | — (não convergiu em 700 eps) | -88.5 | 275.1 | Não |

Checkpoints da média móvel a cada 50 episódios (DQN Completo vs. Sem Target Network):

| Episódio | DQN Completo | Sem Target Network |
|----------|---------------|---------------------|
| 50 | -142.3 | -133.2 |
| 150 | -73.5 | -78.9 |
| 250 | 35.1 | 9.8 |
| 350 | 70.0 | 47.3 |
| 450 | 117.5 | 96.7 |
| 550 | 189.6 | 177.6 |

### Interpretação dos Resultados

**DQN Completo** converge de forma consistente e previsível: a partir do episódio 150 (avg ≈ -73), a média móvel sobe de forma praticamente monotônica em todos os checkpoints de 50 episódios até cruzar o limiar de 200 no episódio 616. O gráfico de recompensa bruta mostra que, a partir de aproximadamente o episódio 400, os pontos passam a se concentrar com mais densidade na faixa de +200 a +300, indicando que a política está se tornando mais consistente, embora ainda ocorram episódios isolados de recompensa fortemente negativa (efeito residual do epsilon-greedy, que nunca cai abaixo de 0.01).

**DQN sem Target Network** apresenta um padrão interessante e didático: durante quase toda a fase intermediária do treinamento (episódios 150 a 550), sua média móvel ficou **atrás** da do DQN Completo em todos os checkpoints (por exemplo, 177.6 vs. 189.6 no episódio 550). Isso é consistente com a teoria: sem target network, o alvo de Bellman se move a cada passo de gradiente, dificultando uma aproximação estável da função Q durante a maior parte do treinamento. No entanto, entre os episódios 550 e 571 esse experimento teve uma alta abrupta (de 177.6 para 200.2 em apenas 21 episódios), cruzando o limiar de solução antes do DQN Completo em termos de contagem absoluta de episódios. Esse resultado pontual não contradiz a teoria — é o tipo de variação que o próprio uso de target network busca reduzir: sem ela, o processo de aprendizado é mais sujeito a "saltos" e desacelerações abruptas (alta variância), em vez do progresso suave observado no experimento completo. Com uma única seed, esse salto tardio é um efeito de variância de uma execução específica, não uma evidência de que a ausência de target network seja vantajosa; o padrão dominante ao longo de quase todo o treinamento foi o oposto (DQN Completo na frente).

**DQN sem Experience Replay** é, sem ambiguidade, o experimento de pior desempenho: a média móvel nunca saiu da faixa negativa (-177 a -50) nos 700 episódios completos, terminando em -88.5 — muito distante do limiar de 200. Diferente dos outros dois experimentos, não há nenhuma tendência de subida sustentada visível no gráfico; a curva oscila em torno de um nível baixo sem progresso direcional. Isso confirma a hipótese teórica: ao treinar em transições sequenciais e correlacionadas (cada gradiente é calculado a partir de uma única transição recém-coletada), a rede sobreajusta repetidamente ao comportamento mais recente e perde a capacidade de reter conhecimento de fases anteriores do episódio (catastrophic forgetting). Vale notar que esse experimento ainda atingiu um pico de recompensa máxima de 275.1 em algum episódio isolado — ou seja, o agente é capaz de pousar bem ocasionalmente — mas não consegue estabilizar essa capacidade ao longo do tempo, exatamente o problema que o replay buffer resolve.

**Avaliação determinística do DQN Completo** (epsilon=0, 10 episódios de teste): média de 194.0 com desvio padrão de 139.5. A média confirma que a política aprendida pousa bem na maior parte das vezes, mas o desvio padrão alto (visível nos episódios individuais: de -115.6 a 303.1) mostra que mesmo a política final, sem exploração, ainda falha ocasionalmente — um lembrete de que "resolver" o ambiente pelo critério de média de 200 ao longo de 100 episódios não elimina toda a variância residual de episódio a episódio.

**Conclusão**: os resultados validam as contribuições do DQN de Mnih et al. (2015), com uma observação importante sobre a robustez estatística de comparações com seed única. O papel do **experience replay** é demonstrado de forma inequívoca: sem ele, o agente não converge dentro do orçamento de episódios, ficando permanentemente na faixa negativa. O papel da **target network** é mais sutil neste experimento específico: ela proporcionou progresso mais consistente e previsível durante a maior parte do treinamento (a métrica relevante para estabilidade, não apenas o episódio exato de cruzamento do limiar), mesmo que o experimento sem target network tenha cruzado o limiar de 200 alguns episódios antes por conta de uma alta tardia. Isso reforça que comparações de RL idealmente deveriam usar múltiplas seeds para reportar médias e desvios — com uma seed única, o "episódio exato de solução" é uma métrica ruidosa, enquanto a trajetória da média móvel ao longo do treinamento é uma evidência mais confiável do papel estabilizador de cada componente.

---

## Requisitos de Instalação

```bash
pip install gymnasium[box2d] torch numpy matplotlib
```

Versão testada: gymnasium >= 0.29, PyTorch >= 2.0, Python 3.10+.

## Execução

Abrir e executar todas as células do notebook `DQN_LunarLander.ipynb` em ordem. O treinamento de 3 experimentos com 700 episódios cada leva aproximadamente 30-90 minutos dependendo do hardware (GPU acelera mas não é obrigatória).
