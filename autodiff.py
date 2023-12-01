import numpy as np


class Node:

    def __init__(self, node_type, inputs, name=None):
        self.node_type = node_type
        self.inputs = inputs
        self.name = name

    def __add__(self, other):
        return _num_wrapper(add, self, other)

    def __radd__(self, other):
        return _num_wrapper(add, other, self)

    def __sub__(self, other):
        return _num_wrapper(sub, self, other)

    def __rsub__(self, other):
        return _num_wrapper(sub, other, self)

    def __mul__(self, other):
        return _num_wrapper(mul, self, other)

    def __rmul__(self, other):
        return _num_wrapper(mul, other, self)

    def return_children(self):
        '''
        Returns children of node
        '''
        children = set()

        def _dfs(node):

            if isinstance(node.inputs, int) or isinstance(node.inputs, float) or isinstance(node.inputs, np.ndarray):
                children.add(node)
                return

            else:
                children.add(node)
                for inp in node.inputs:
                    _dfs(inp)

        _dfs(self)

        return children

    def _topological_sort(self):
        '''
        Returns topological sort for graph comprised of head node
        and its children
        '''
        ordering = []

        vis = set()

        def _dfs(node):

            if node not in vis:
                vis.add(node)
                if isinstance(node, Operator):
                    for inp in node.inputs:
                        _dfs(inp)
                ordering.append(node)

        _dfs(self)

        return ordering

    def set_ordering(self):
        self.ordering = self._topological_sort()

    def backward_pass(self):
        '''Performs backward pass.

        self.set_ordering() must be called first
        '''

        if not hasattr(self, 'ordering'):
            raise Exception('ordering must be defined by performing backward pass')

        vis = set()
        self.ordering[-1].gradient = 1
        for node in reversed(self.ordering):

            if isinstance(node, Operator):
                inputs = node.inputs
                grads = node.backward(dout=node.gradient)

                for inp, grad in zip(inputs, grads):

                    if inp not in vis:
                        inp.gradient = grad
                    else:
                        inp.gradient += grad
                    vis.add(inp)

        return {node.name: node.gradient for node in self.ordering}


class Constant(Node):
    count = 0

    def __init__(self, inputs, name=None):
        self.inputs = inputs
        self.name = f'c{Constant.count}' if name is None else name
        super().__init__('constant', self.inputs, self.name)

        self.value = inputs

        self.gradient = None

        Constant.count += 1

    def __repr__(self):
        return f'{self.name}/{self.inputs}'


class Variable(Node):
    count = 0

    def __init__(self, inputs, name=None):
        self.inputs = inputs
        self.name = f'v{Variable.count}' if name is None else name
        super().__init__('variable', self.inputs, self.name)

        self.value = inputs

        self.gradient = None

        Variable.count += 1

    def __repr__(self):
        return f'{self.name}/{self.inputs}'


class Operator(Node):

    def __init__(self, inputs, name=None):
        self.inputs = inputs
        self.name = name
        super().__init__('operator', self.inputs, self.name)

    def __repr__(self):
        return f'{self.name}/{[inp.name for inp in self.inputs]}'


class add(Operator):
    count = 0

    def __init__(self, p1, p2, name=None):

        self.p1 = p1
        self.p2 = p2
        self.inputs = [self.p1, self.p2]
        self.name = f'add{add.count}' if name is None else name

        self.value = self.p1.value + self.p2.value

        super().__init__(self.inputs, self.name)

        add.count += 1

    def backward(self, dout):

        if isinstance(dout, int) or isinstance(dout, float):
            return dout, dout

        # handle broadcasted tensors
        sum_axes1 = ()
        sum_axes2 = ()
        for ax in range(self.p1.value.ndim):
            if self.p1.value.shape[ax] == 1:
                sum_axes1 += (ax,)
            elif self.p2.value.shape[ax] == 1:
                sum_axes2 += (ax,)

        return dout.sum(axis=sum_axes1), dout.sum(axis=sum_axes2)


class sub(Operator):

    count = 0

    def __init__(self, p1, p2, name=None):
        self.p1 = p1
        self.p2 = p2
        self.inputs = [self.p1, self.p2]
        self.name = f'sub{sub.count}' if name is None else name

        super().__init__(self.inputs, self.name)

        self.value = self.p1.value - self.p2.value

        sub.count += 1

    def backward(self, dout):

        if isinstance(dout, int) or isinstance(dout, float):
            return dout, dout

        # handle broadcast tensors
        p1_sum_axes = ()
        p2_sum_axes = ()
        for ax in range(self.p1.value.ndim):
            if self.p1.value.shape[ax] > self.p2.value.shape[ax]:
                p2_sum_axes += (ax,)
            elif self.p1.value.shape[ax] < self.p2.value.shape[ax]:
                p1_sum_axes += (ax,)

        return dout.sum(axis=p1_sum_axes), -dout.sum(axis=p2_sum_axes)


class mul(Operator):
    count = 0

    def __init__(self, p1, p2, name=None):
        self.p1 = p1
        self.p2 = p2
        self.inputs = [p1, p2]
        self.name = f'mul{mul.count}' if name is None else name

        super().__init__(self.inputs, self.name)

        self.value = self.p1.value * self.p2.value

        mul.count += 1

    def backward(self, dout):
        return dout * self.p2.value, dout * self.p1.value


class tensorcontract(Operator):
    count = 0

    def __init__(self, p1, p2, axes, name=None):
        self.p1 = p1
        self.p2 = p2
        self.inputs = [p1, p2]
        self.name = f'tensorcontract{tensorcontract.count}' if name is None else name

        super().__init__(self.inputs, self.name)

        self.forward_axes = axes
        self.p1_forward_axes = axes[0]
        self.p2_forward_axes = axes[1]

        self.value = np.tensordot(self.p1.value, self.p2.value, self.forward_axes)

        tensorcontract.count += 1

    def backward(self, dout):
        if isinstance(dout, int) or isinstance(dout, float):
            return dout * self.p2.value, dout * self.p1.value

        else:
            p1_backward_axes = [i for i in range(0, self.p1.value.ndim) if i not in self.p1_forward_axes]
            p2_backward_axes = [i for i in range(0, self.p2.value.ndim) if i not in self.p2_forward_axes]

            dout_axes = [i for i in range(0, dout.ndim)]

            num_p1_backward = len(p1_backward_axes)
            num_p2_backward = len(p2_backward_axes)

            p1_grad = np.tensordot(dout, self.p2.value, axes=(dout_axes[-num_p2_backward:], p2_backward_axes))
            p2_grad = np.tensordot(dout, self.p1.value, axes=(dout_axes[:num_p1_backward], p1_backward_axes))

            p1_undo_mapping = index_mapper(self.p1_forward_axes, self.p2_forward_axes, self.p1.value.ndim)
            p2_undo_mapping = index_mapper(self.p2_forward_axes, self.p1_forward_axes, self.p2.value.ndim)


            return p1_grad.transpose(p1_undo_mapping), p2_grad.transpose(p2_undo_mapping)


class logsoft(Operator):

    count = 0

    def __init__(self, p1, name=None):
        '''
        Numerically stable logarithm of softmax

        :param p1: node whose value is n x q matrix of logits, for n observations with q possible classes

        value: value is n x q matrix of log probabilities
        '''
        self.p1 = p1
        self.inputs = [p1]
        self.name = f'logsoft{logsoft.count}' if name is None else name

        super().__init__(self.inputs, self.name)

        self.maxs = self.p1.value.max(axis=1, keepdims=True)

        self.value = self.p1.value - self.maxs - np.log(np.exp(self.p1.value - self.maxs).sum(axis=1, keepdims=True))

        logsoft.count += 1

    def backward(self, dout):
        Z = np.exp(self.p1.value).sum(axis=1, keepdims=True)
        return [dout - dout.sum(axis=1, keepdims=True) * np.exp(self.p1.value) / Z]


class cross_entropy(Operator):

    count = 0

    def __init__(self, classes, p1, name=None):
        '''
        Computes cross entropy

        :param classes: (n,) numpy array where each entry is true class of that observation
        :param p1: Node whose value is n x q matrix of log probabilities
        '''
        self.classes = classes
        self.p1 = p1
        self.inputs = [p1]
        self.name = f'cross_entropy{cross_entropy.count}' if name is None else name

        super().__init__(self.inputs, self.name)

        self.value = -p1.value[[i for i in range(p1.value.shape[0])], self.classes].sum()

        cross_entropy.count += 1

    def backward(self, dout):
        p1_grad = np.zeros((self.p1.value.shape[0], self.p1.value.shape[1]))
        p1_grad[[i for i in range(self.p1.value.shape[0])], self.classes] = -1

        return [dout * p1_grad]


def _num_wrapper(func, a, b):
    '''
    Wraps integers, floats, and numpy arrays in Constant class
    '''
    if isinstance(a, int) or isinstance(a, float) or isinstance(a, np.ndarray):
        return func(Constant(a), b)

    elif isinstance(b, int) or isinstance(b, float) or isinstance(b, np.ndarray):
        return func(a, Constant(b))

    else:
        return func(a, b)


def index_mapper(A_forward, B_forward, num_A_ind):

    A_forward_comp = [i for i in range(num_A_ind) if i not in A_forward]

    total_mapping = {}
    for index, entry in enumerate(A_forward_comp):
        total_mapping[entry] = index

    ff_mapping = [[A_forward[i], B_forward[i]] for i in range(len(A_forward))]
    ff_mapping_sort = sorted(ff_mapping, key=lambda x: x[1])

    for index, pair in enumerate(ff_mapping_sort):
        total_mapping[pair[0]] = index + len(A_forward_comp)

    undo_map_list = [total_mapping[i] for i in range(len(total_mapping))]

    return undo_map_list

print('test')