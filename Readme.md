# 基于硬件对齐的多级注意力分组压缩推理优化

## 概述

Qwen2.5是阿里云研发的通义千问系列开源大模型，2024年9月推出的系列涵盖0.5B到72B参数规模的语言模型、多模态模型及专业领域模型。

该系列包含Base、Instruct等多个版本，支持128K tokens上下文处理与8K tokens内容生成，覆盖中文、英文、法文等29种以上语言 。其基于18万亿Token数据集训练，在[MMLU](https://baike.baidu.com/item/MMLU/64566893?fromModule=lemma_inlink)、HumanEval、MATH评测中分别达到85+、85+、80+评分，适配函数计算FC、人工智能平台PAI及GPU云服务器等部署方案。Qwen系列累计下载量超4000万次，衍生模型数量达7.8万个。引入时间对齐的多模态ROPE技术，显著强化对长视频序列的时序理解能力。

本样例基于transformers库完成对于qwen2.5-7b量化模型的单卡适配加速推理

## 支持的产品

Atlas A3 系列产品

## 环境准备

1. 安装CANN软件包

   本样例的编译执行依赖CANN开发套件包（cann-toolkit）与CANN二进制算子包（cann-kernels），支持的CANN软件版本为`8.2.RC1`

   查看CPU架构：

   ```
   lscpu  
   # aarch64/X86_64
   ```

   [社区版资源下载-资源下载中心-昇腾社区](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.2.RC1)

   选择8.2RC1版本(推荐)：` Ascend-cann-toolkit_8.2.RC1_linux-aarch64.run`并且放置到服务器。

   ```shell
   sftp ilisa415416@222.197.166.125             #连接SSH的SFTP服务
   put Ascend-cann-toolkit_8.2.RC1_linux-aarch64.run   #上传文件
   ```

   以非root用户HwHiAiUser（驱动固件的默认运行用户）安装CANN软件。需要准备用户组：

   ```shell
   sudo usermod -G HwHiAiUser ilisa415416
   ```

   准备安装：

   ```shell
   # 添加可执行权限
   chmod +x Ascend-cann-toolkit_6.3.RC1_linux-x86_64.run
   # 校验软件包的一致性和完整性
   ./Ascend-cann-toolkit_6.3.RC1_linux-x86_64.run --check
   # 执行安装命令
   ./Ascend-cann-toolkit_6.3.RC1_linux-x86_64.run --install --install-for-all
   ```

   显示如下信息，则表示安装完成：

   ```
   xxx install success
   ```

   默认安装路径为：

   ```
   /home/name/Ascend
   ```

   设置环境变量：

   ```shell
   add "source /home/name/Ascend/ascend-toolkit/set_env.sh" to ~/.bashrc.
   source ~/.bashrc    #生效
   ```

2. 安装Ascend Extension for PyTorch（torch_npu）

   请从[软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.3.RC1.alpha002)下载`Ascend-cann-toolkit_${version}_linux-${arch}.run`与`Atlas-A3-cann-kernels_${version}_linux-${arch}.run`软件包，并参考[CANN安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/softwareinst/instg/instg_0001.html?Mode=PmIns&OS=Debian&Software=cannToolKit)进行安装。

   - `${version}`表示CANN包版本号，如`8.2.RC1`。
   - `${arch}`表示CPU架构，如aarch64。

   **Ascend Extension for PyTorch**的分支名称采用`{PyTorch版本}-{昇腾版本}`命名规则，前者为**Ascend Extension for PyTorch**匹配的PyTorch版本，后者为**Ascend Extension for PyTorch**版本号，详细匹配如下：

   | CANN版本       | 支持的PyTorch版本 | 支持的Extension版本 | Gitee分支         |
   | -------------- | ----------------- | ------------------- | ----------------- |
   | CANN 8.2.RC1   | 2.6.0             | 2.6.0               | v2.6.0-7.1.0      |
   |                | 2.5.1             | 2.5.1.post1         | v2.5.1-7.1.0      |
   |                | 2.1.0             | 2.1.0.post13        | v2.1.0-7.1.0      |
   | CANN 8.1.RC1   | 2.5.1             | 2.5.1               | v2.5.1-7.0.0      |
   |                | 2.4.0             | 2.4.0.post4         | v2.4.0-7.0.0      |
   |                | 2.3.1             | 2.3.1.post6         | v2.3.1-7.0.0      |
   |                | 2.1.0             | 2.1.0.post12        | v2.1.0-7.0.0      |
   | CANN 8.0.0     | 2.4.0             | 2.4.0.post2         | v2.4.0-6.0.0      |
   |                | 2.3.1             | 2.3.1.post4         | v2.3.1-6.0.0      |
   |                | 2.1.0             | 2.1.0.post10        | v2.1.0-6.0.0      |
   | CANN 8.0.RC3   | 2.4.0             | 2.4.0               | v2.4.0-6.0.rc3    |
   |                | 2.3.1             | 2.3.1.post2         | v2.3.1-6.0.rc3    |
   |                | 2.1.0             | 2.1.0.post8         | v2.1.0-6.0.rc3    |
   | CANN 8.0.RC2   | 2.3.1             | 2.3.1               | v2.3.1-6.0.rc2    |
   |                | 2.2.0             | 2.2.0.post2         | v2.2.0-6.0.rc2    |
   |                | 2.1.0             | 2.1.0.post6         | v2.1.0-6.0.rc2    |
   |                | 1.11.0            | 1.11.0.post14       | v1.11.0-6.0.rc2   |
   | CANN 8.0.RC1   | 2.2.0             | 2.2.0               | v2.2.0-6.0.rc1    |
   |                | 2.1.0             | 2.1.0.post4         | v2.1.0-6.0.rc1    |
   |                | 1.11.0            | 1.11.0.post11       | v1.11.0-6.0.rc1   |
   | CANN 7.0.0     | 2.1.0             | 2.1.0               | v2.1.0-5.0.0      |
   |                | 2.0.1             | 2.0.1.post1         | v2.0.1-5.0.0      |
   |                | 1.11.0            | 1.11.0.post8        | v1.11.0-5.0.0     |
   | CANN 7.0.RC1   | 2.1.0             | 2.1.0.rc1           | v2.1.0-5.0.rc3    |
   |                | 2.0.1             | 2.0.1               | v2.0.1-5.0.rc3    |
   |                | 1.11.0            | 1.11.0.post4        | v1.11.0-5.0.rc3   |
   | CANN 6.3.RC3.1 | 1.11.0            | 1.11.0.post3        | v1.11.0-5.0.rc2.2 |
   | CANN 6.3.RC3   | 1.11.0            | 1.11.0.post2        | v1.11.0-5.0.rc2.1 |
   | CANN 6.3.RC2   | 2.0.1             | 2.0.1.rc1           | v2.0.1-5.0.rc2    |
   |                | 1.11.0            | 1.11.0.post1        | v1.11.0-5.0.rc2   |
   |                | 1.8.1             | 1.8.1.post2         | v1.8.1-5.0.rc2    |

   建议进行源码编译安装，以适配后续自定义算子等操作。

3. 下载项目源码并安装依赖的python库

   ```shell
   # 下载项目源码，以master分支为例
   git clone ######
   
   # 安装依赖的python库 ，仅支持python3.9.9
   cd qwen2.5-compression/
   pip3 install -r .requirements.txt
   ```


## 权重准备

请根据所使用的模型类型自行下载原始权重到本地路径，例如`./models/qwen2.5/`。

代码脚本在执行过程中会自动检查对应模型路径，若没有则会自动下载。

## 推理执行

执行如下命令即可拉起单卡推理任务

```bash
bash infer.sh
```


## 前端显示
打开浏览器输入`localhost:port`进行对话

