# 不绑定 Codex，也能完成 GitHub -> Hugging Face 自动训练发布

## 一、项目准备

本项目已包含你提供的 `imdb_top_500.csv`，不需要另找数据。
采用老师推荐的 TF-IDF + 小型前馈神经网络方案。

解压项目包后，仓库根目录应直接出现 `train.py`、`predict.py`、`requirements.txt`
和 `.github` 文件夹，不要把整个项目再嵌套在另一层文件夹里。

## 二、通过 GitHub 网页上传代码

1. 登录 GitHub，新建仓库，建议名称 `imdb-sentiment-nn`。
2. 进入仓库，选择 Add file -> Upload files。
3. 将解压后的项目文件和文件夹拖到页面中，并提交到 `main` 分支。
4. 检查仓库确实包含以下路径：

   - `data/imdb_top_500.csv`
   - `.github/workflows/train-and-upload.yml`
   - `train.py`
   - `predict.py`
   - `sentiment.py`
   - `upload_to_hf.py`
   - `requirements.txt`
   - `tests/test_pipeline.py`

不要直接上传 ZIP 代替源码。特别注意 `.github` 以点开头，可能被文件管理器隐藏。
若网页拖拽没有带上它，可用 Add file -> Create new file，文件名输入
`.github/workflows/train-and-upload.yml`，粘贴项目中对应 YAML 的全部内容并提交。

## 三、设置 Hugging Face Token（这一步你在自己的账号中操作）

1. 登录 Hugging Face。可先创建空的 **Model** 仓库，例如 `imdb-sentiment-nn`。
2. 在 Hugging Face 的 Access Tokens 页面创建具有该仓库写入权限的 Token。
   若希望工作流自动创建仓库，Token 还需要相应的创建权限。
3. 在 GitHub 项目中打开 Settings -> Secrets and variables -> Actions。
4. 在 Secrets 页选择 New repository secret：

   - Name: `HF_TOKEN`
   - Secret: 你的 Hugging Face Token

5. 可选：在 Variables 页添加 `HF_REPO_ID`，值为 `你的HF用户名/imdb-sentiment-nn`。
   不设置时，程序自动使用 Token 所属用户名和 GitHub 仓库名。

Token 仅填入 GitHub Secret，不要发到聊天中，也不要写进 YAML 或 Python 文件。

## 四、运行并核对

1. 打开仓库 Actions 页，选择 Train and Upload to Hugging Face。
2. 点击 Run workflow，选择 main（或 master），再运行。
3. 流程应依次完成测试、训练、预测验证、保存产物、上传模型。
4. 在 Summary 查看真实测试准确率、F1 和 Hugging Face 仓库链接。
5. 打开 HF 仓库的 Files 页面，核对 `model.pt`、`vectorizer.pkl`、
   `config.json`、`metrics.json` 已出现，并检查模型卡上的 GitHub commit。

首次上传代码时没有设置 Token，最后一步会失败；设置好后重新运行即可。
不要只看见模型压缩包就认为部署完成，最后的 HF 上传步骤也必须成功。

## 五、按老师要求提交

在上述整条流程成功后，将 GitHub 仓库 URL 和 Hugging Face 模型仓库 URL
发送到课程微信群。本项目不会替你向微信群发送消息。

后续修改仅提交 GitHub，让 Actions 重新训练并自动更新 HF；不手动上传模型。
本地验证生成的模型用于检查代码，不作为提交时的部署模型。

## 六、常见失败

| 现象 | 检查项 |
|---|---|
| Actions 没有工作流 | YAML 是否位于仓库根目录下的 `.github/workflows/` |
| 提交后不运行 | 分支名是否为 `main` 或 `master`，也可手动 Run workflow |
| 找不到 CSV | 文件是否在 `data/imdb_top_500.csv`，不要把整个项目嵌套一层 |
| 最后提示缺少 HF_TOKEN | 应创建 Actions Secret，名称大小写必须完全一致 |
| HF 返回 401/403 | Token 是否有效、是否有目标仓库的写权限，HF_REPO_ID 是否正确 |
| 已存在仓库创建仍报权限问题 | 确认目标是你有权限的 Model 仓库，必要时先手动创建空仓库 |
| 有模型文件但不能网页推理 | 本作业是 Hub 文件发布，不要求部署在线网页或推理服务 |

## 七、数据与结果的解释

500 条样本中正负各 250 条，默认分成 350/75/75（清洗去重后如有变化会记录）。
只使用 `text`，不用 `rating`。TF-IDF 只在训练集拟合；验证集选择最佳模型；
最后只对测试集做一次最终评价。测试集很小，不应把一次准确率当作广泛泛化能力。
