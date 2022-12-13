import onnx

model = onnx.load('gpt2-optimized.onnx')
for n in model.graph.node:
    for inp in n.input:
        if inp == '3479':
            print(n, inp)
for i in model.graph.initializer:
    if i.name == '3479':
        print('initialize', i)

ops = set()
for n in model.graph.node:
    ops.add(n.op_type)
print(ops)