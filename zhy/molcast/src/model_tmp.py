class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_feats, n_hidden, 'mean'))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        # block 是我们采样获得的二部图，这里用于消息传播
        # x 为节点特征
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h_dst = h[:block.number_of_dst_nodes()]
            h = layer(block, (h, h_dst))
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, x, batch_size, device):
        # inference 用于评估测试，针对的是完全图
        # 目前会出现重复计算的问题，优化方案还在 to do list 上
        nodes = torch.arange(g.number_of_nodes())
        for l, layer in enumerate(self.layers):
            y = torch.zeros(g.number_of_nodes(),
                            self.n_hidden if l != len(self.layers) - 1 else self.n_classes)
            for start in tqdm.trange(0, len(nodes), batch_size):
                end = start + batch_size
                batch_nodes = nodes[start:end]
                block = dgl.to_block(dgl.in_subgraph(
                    g, batch_nodes), batch_nodes)
                input_nodes = block.srcdata[dgl.NID]
                h = x[input_nodes].to(device)
                h_dst = h[:block.number_of_dst_nodes()]
                h = layer(block, (h, h_dst))
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)
                y[start:end] = h.cpu()
            x = y
        return y


def compute_acc(pred, labels):
    """
    计算准确率
    """
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)


def evaluate(model, g, inputs, labels, val_mask, batch_size, device):
    """
    评估模型，调用 model 的 inference 函数
    """
    model.eval()
    with torch.no_grad():
        pred = model.inference(g, inputs, batch_size, device)
    model.train()
    return compute_acc(pred[val_mask], labels[val_mask])


def load_subtensor(g, labels, seeds, input_nodes, device):
    """
    将一组节点的特征和标签复制到 GPU 上。
    """
    batch_inputs = g.ndata['features'][input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels

# 近邻采样


class NeighborSampler(object):
    def __init__(self, g, fanouts):
        """
        g 为 DGLGraph；
        fanouts 为采样节点的数量，实验使用 10,25，指一阶邻居采样 10 个，二阶邻居采样 25 个。
        """
        self.g = g
        self.fanouts = fanouts

    def sample_blocks(self, seeds):
        seeds = th.LongTensor(np.asarray(seeds))
        blocks = []
        for fanout in self.fanouts:
            # sample_neighbors 可以对每一个种子的节点进行邻居采样并返回相应的子图
            # replace=True 表示用采样后的邻居节点代替所有邻居节点
            frontier = dgl.sampling.sample_neighbors(
                g, seeds, fanout, replace=True)
            # 将图转变为可以用于消息传递的二部图（源节点和目的节点）
            # 其中源节点的 id 也可能包含目的节点的 id（原因上面说了）
            # 转变为二部图主要是为了方便进行消息传递
            block = dgl.to_block(frontier, seeds)
            # 获取新图的源节点作为种子节点，为下一层作准备
            # 之所以是从 src 中获取种子节点，是因为采样操作相对于聚合操作来说是一个逆向操作
            seeds = block.srcdata[dgl.NID]
            # 把这一层放在最前面。
            # PS：如果数据量大的话，插入操作是不是不太友好。
            blocks.insert(0, block)
        return blocks

?dgl.sampling.sample_neighbors
b = dgl.to_block(g)

?dgl.NID



# 参数设置
gpu = -1
num_epochs = 20
num_hidden = 16
num_layers = 2
fan_out = '10,25'
batch_size = 1000
log_every = 20  # 记录日志的频率
eval_every = 5
lr = 0.003
dropout = 0.5
num_workers = 0  # 用于采样进程的数量

if gpu >= 0:
    device = torch.evice('cuda:%d' % gpu)
else:
    device = torch.device('cpu')

# load reddit data
# NumNodes: 232965
# NumEdges: 114848857
# NumFeats: 602
# NumClasses: 41
# NumTrainingSamples: 153431, first 20 days
# NumValidationSamples: 23831
# NumTestSamples: 55703,  remaining days

data = RedditDataset(self_loop=True)
# ?RedditDataset
train_mask = data.train_mask
# data.features.shape: torch.Size([232965, 602])
val_mask = data.val_mask
features = torch.Tensor(data.features)
in_feats = features.shape[1]
labels = data.labels
n_classes = data.num_labels
# Construct graph
g = data[0]

# 已知部分节点的label,预测另外部分节点的label

################################

# 开始训练
train_nid = th.LongTensor(np.nonzero(train_mask)[0])
val_nid = th.LongTensor(np.nonzero(val_mask)[0])
train_mask = th.BoolTensor(train_mask)
val_mask = th.BoolTensor(val_mask)

# Create sampler
sampler = NeighborSampler(g, [int(fanout) for fanout in fan_out.split(',')])

# Create PyTorch DataLoader for constructing blocks
# collate_fn 参数指定了 sampler，可以对 batch 中的节点进行采样
dataloader = DataLoader(
    dataset=train_nid.numpy(),
    batch_size=batch_size,
    collate_fn=sampler.sample_blocks,
    shuffle=True,
    drop_last=False,
    num_workers=num_workers)

# Define model and optimizer
model = GraphSAGE(in_feats, num_hidden, n_classes, num_layers, F.relu, dropout)
model = model.to(device)
loss_fcn = nn.CrossEntropyLoss()
loss_fcn = loss_fcn.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
avg = 0
iter_tput = []
for epoch in range(num_epochs):
    tic = time.time()

    for step, blocks in enumerate(dataloader):
        tic_step = time.time()

        input_nodes = blocks[0].srcdata[dgl.NID]
        seeds = blocks[-1].dstdata[dgl.NID]

        # Load the input features as well as output labels
        batch_inputs, batch_labels = load_subtensor(
            g, labels, seeds, input_nodes, device)

        # Compute loss and prediction
        batch_pred = model(blocks, batch_inputs)
        loss = loss_fcn(batch_pred, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iter_tput.append(len(seeds) / (time.time() - tic_step))
        if step % log_every == 0:
            acc = compute_acc(batch_pred, batch_labels)
            gpu_mem_alloc = th.cuda.max_memory_allocated(
            ) / 1000000 if th.cuda.is_available() else 0
            print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MiB'.format(
                epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:]), gpu_mem_alloc))

    toc = time.time()
    print('Epoch Time(s): {:.4f}'.format(toc - tic))
    if epoch >= 5:
        avg += toc - tic
    if epoch % eval_every == 0 and epoch != 0:
        eval_acc = evaluate(
            model, g, g.ndata['features'], labels, val_mask, batch_size, device)
        print('Eval Acc {:.4f}'.format(eval_acc))

print('Avg epoch time: {}'.format(avg / (epoch - 4)))