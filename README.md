# chart_maker

Gera automaticamente charts de **Clone Hero** com **guitarra e bateria no Expert** a partir de um arquivo de música (mp3, ogg, flac, wav, m4a…).

Você envia a música por uma página web local (ou pela linha de comando) e recebe uma pasta pronta para o jogo:

```
Artista - Música/
├── notes.chart   # guitarra (ExpertSingle) e bateria pro (ExpertDrums)
├── song.ini      # metadados, dificuldade, pro_drums = True
├── song.ogg      # áudio (com alguns segundos de silêncio no início para dar tempo de ver as notas)
└── album.png     # capa embutida no arquivo, a que você enviar, ou uma gerada com o nome da música
```

## Instalação

Requisitos: Python 3.12+, [uv](https://docs.astral.sh/uv/) e ffmpeg (`sudo apt install ffmpeg`).

```bash
uv sync
```

Isso instala o PyTorch na versão só-CPU (não precisa de placa NVIDIA) e ocupa ~1,5 GB. Na primeira música, os modelos são baixados da internet (~130 MB, ficam em cache em `~/.cache`).

## Uso

### Pela página web

```bash
uv run chart-maker serve --songs-dir "/caminho/para/Clone Hero/Songs"
```

Abra http://127.0.0.1:8000, arraste a música e clique em **Gerar chart**. Quando terminar dá para:

- **Pré-visualizar** o chart no navegador, com a pista rolando junto com a música;
- **Baixar .zip** da pasta da música;
- **Instalar no Clone Hero** com um clique (se você passou `--songs-dir`).

`--songs-dir` é a pasta configurada no Clone Hero em *Settings → General → Song Folders*. Depois de instalar, faça *Scan Songs* no jogo. Sem `--songs-dir`, os charts ficam em `./output`.

### Pela linha de comando

```bash
uv run chart-maker gerar musica.mp3 -o "/caminho/para/Clone Hero/Songs"
uv run chart-maker gerar *.mp3 --stems          # várias músicas, com áudio separado por instrumento
```

| Opção | Para que serve |
| --- | --- |
| `-o PASTA` | Onde criar a pasta da música (padrão `./output`, ou a variável `CHART_MAKER_OUTPUT`). |
| `--title`, `--artist`, `--album`, `--genre`, `--year` | Metadados. Por padrão uso as tags do arquivo. |
| `--cover IMAGEM` | Capa do álbum. |
| `--stems` | Grava `guitar.ogg` e `drums.ogg` separados: no jogo o instrumento fica mudo quando você erra. |
| `--guitar-source` | De onde tirar a guitarra: `auto` (padrão), `guitar`, `other`, `piano`, `guitar+other`, `guitar+piano`. Útil quando a música não tem guitarra ou a separação errou. |
| `--no-cache` | Não reaproveita a separação de instrumentos de uma execução anterior. |
| `--device cuda` | Usa a GPU (NVIDIA) se o PyTorch com CUDA estiver instalado. |

Leva de 1 a 4 minutos por música em CPU; quase todo esse tempo é a separação de instrumentos. Ela fica em cache (as 3 últimas músicas), então gerar de novo a mesma música leva uns 15 segundos.

## Como funciona

1. **Separação de instrumentos** com [Demucs](https://github.com/adefossez/demucs) (`htdemucs_6s`): bateria, guitarra, baixo, voz, piano e "outros".
2. **Andamento e compassos** com [Beat This!](https://github.com/CPJKU/beat_this). As ativações do modelo passam por programação dinâmica (tempo estável) e por um HMM de compassos. O mapa de tempo usa um único BPM quando a música foi gravada no metrônomo e vários quando o andamento varia. A grade é alinhada pelos ataques do bumbo e da caixa.
3. **Bateria** com [ADTOF](https://github.com/MZehren/ADTOF) ([port em PyTorch](https://github.com/xavriley/ADTOF-pytorch)), modelo treinado com charts de jogos de ritmo: bumbo, caixa, tons, chimbal e pratos. Depois disso:
   - os tons são divididos em amarelo/azul/verde pela altura;
   - pratos em padrão contínuo viram condução (azul) e os isolados viram ataque (verde);
   - ficam no máximo duas peças de mão por vez, e tom e prato da mesma cor não caem juntos;
   - bumbos rápidos alternam com notas de 2x kick (Expert+).
4. **Guitarra**: ataques por fluxo espectral, altura por soma harmônica numa transformada de Q constante e acordes pela energia de quintas e terças acima do ruído. Os trastes seguem o contorno melódico: a mesma nota fica no mesmo traste e uma nota mais aguda vai para a direita. Notas longas viram sustain.
5. **Chart**:
   - as notas são encaixadas em semicolcheias ou tercinas, decidido batida a batida;
   - star power aparece a cada ~25 s e as viradas de ativação da bateria caem antes de ataques de prato;
   - as seções do modo prática vêm da segmentação estrutural da música.

O `notes.chart` gerado é conferido por um validador interno. Durante o desenvolvimento, os charts também foram verificados com o [scan-chart](https://github.com/Geomitron/scan-chart), parser usado pelo Chorus Encore que reproduz o do Clone Hero.

## Limitações

- Chart automático não substitui um chart feito à mão. Notas extras ou faltando, acordes no lugar de notas simples e trastes estranhos acontecem, principalmente em mixagens densas. Para corrigir, abra o `notes.chart` no [Moonscraper](https://github.com/FireFox2000000/Moonscraper-Chart-Editor).
- Funciona melhor em rock/pop com bateria acústica e guitarra bem audível. Em músicas sem guitarra, o instrumento principal (synth, piano) vira a guitarra.
- Só a dificuldade Expert é gerada.

## Desenvolvimento

```bash
uv run pytest            # testes rápidos
uv run pytest -m slow    # ponta a ponta com uma música sintética (roda os modelos)
```

Código em `src/chart_maker/`: `pipeline.py` orquestra tudo; `tempo.py`, `drums.py`, `guitar.py`, `quantize.py`, `phrases.py` e `sections.py` fazem a análise; `chart.py` e `songini.py` escrevem os arquivos; `web/` é a interface.

## Licenças dos modelos

Demucs e Beat This! são MIT. Os pesos do ADTOF são **CC BY-NC-SA 4.0** (uso não comercial).
