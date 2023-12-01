# Autodifferentiator

Simple autodifferentiator package built from scratch.  Currently contains operations for:
- Addition / subtraction / multiplication of scalars and numpy arrays 
- Arbitrary contractions of two tensors
- Numerically stable logarithm of softmax 
- Cross entropy

Basically, it includes everything you would need for logistic regression. _Classifier.ipynb_ is a jupyter notebook
showing a few test cases. It culminates by using the autodifferentiator in a logistic regression model trained on the 
[FashionMNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist) dataset.



## Tricky operations in the backward pass

### Tensor Contractions
The backward pass for the __tensorcontract__ operation takes a little work to get right. Start with the example

$$\frac{\partial \mathcal{L}}{\partial A_{r_0 r_1 r_2 r_3}} = \frac{\partial \mathcal{L}}{\partial C_{j_0 j_1 j_2 j_3}}\frac{\partial C_{j_0 j_1 j_2 j_3}}{\partial A_{r_0 r_1 r_2 r_3}}. $$ For convenience define

$$ Q_{j_0 j_1 j_2 j_3} \equiv \frac{\partial \mathcal{L}}{\partial C_{j_0 j_1 j_2 j_3}}.$$

For the sake of having a concrete example to work off of, we'll take

$$C_{j_0 j_1 j_2 j_3} = A_{d_0 j_0 d_1 j_1}B_{j_2 d_1 d_0 j_3}. $$ 

$A$ is contracted with $B$ along $A_{\text{forward}} = [0,2]$ and $B_{\text{forward}} = [2, 1]$, and we have defined these objects so that the ordering of $B_{\text{forward}}$ matches that of $A_{\text{forward}}$. We can compute the derivative and contract $\delta$'s to arrive at

$$Q_{r_1 r_3 j_2 j_3}B_{j_2 r_2 r_0 j_3}.$$

Observe that we contract $Q$ with $B$ along the _compliment_ of $B_{\text{forward}}$, namely $B_{\text{backward}} = [0,3]$ (this time defined with entries in increasing order). Moreover, we contract the last $\textbf{len}(B_{\text{backward}}) = 2$ indices of $Q$ with $B$.

The last thing to notice is that if we compute this product using __np.tensordot__, it will return a tensor with dimensions $r_1 \times r_3 \times r_2 \times r_0$, whereas we want a tensor with dimensions $r_0 \times r_1 \times r_2 \times r_3$. This means that at the end, we need to transpose the axes to the correct ordering, so we need to keep track of how they are permuted through our computation.

We write a helper function __index_mapper__ to help us do this, which accepts $A_{\text{forward}}$, $B_{\text{forward}}$, and num_A_ind (the number of $A$ axes) as arguments.  Conceptually, we associate each $r$ with an index of $A$, namely

$$
(r_0, d_0), (r_1, j_0), (r_2, d_1), (r_3, j_1)
$$

Indicies associated with $j$'s end up on $Q$. Because the $j$'s on $Q_{j_0 j_1 j_2 j_3}$ are ordered, the ordering of $r$'s on $Q$ is determined by its corresponding $j$. In our example, this means the ordering we end up with is 

$$(r_1, j_0), (r_3, j_1).$$

The code block which does this is



    def index_mapper(A_forward, B_forward, num_A_ind):

        A_forward_comp = [i for i in range(num_A_ind) if i not in A_forward]
    
        total_mapping = {}
        for index, entry in enumerate(A_forward_comp):
            total_mapping[entry] = index
            
    


where total_mapping is dictionary that tracks each $r$'s location in the final ordering. At this point, it would read $\{1:0, 3:1\}$.  

Next, the $r$'s that are associated with $d$'s end up on $B$, with their position dictated by the mapping $A_{\text{forward}} = [0,2]\rightarrow B_{\text{forward}} = [2, 1]$.  More specifically, we get

$$(r_2, d_1), (r_0, d_0).$$

This is because the $r_0$ is mapped to $2$ in $B_{\text{forward}}$, and so it should come after $r_1$, which is mapped to $1$. This code which does this is 



    ff_mapping = [[A_forward[i], B_forward[i]] for i in range(len(A_forward))]
    ff_mapping_sort = sorted(ff_mapping, key=lambda x: x[1])

    for index, pair in enumerate(ff_mapping_sort):
        total_mapping[pair[0]] = index + len(A_forward_comp)



Essentially, we form pairs of $A_{\text{forward}}$ and $B_{\text{forward}}$ values, and sort by the latter. Finally, because we perform __np.tensordot__ with $Q$ on the left, we offset by len(A_forward_comp). Overall, we end up with 

$$(r_1, j_0), (r_3, j_1), (r_2, d_1), (r_0, d_0),$$

and our dictionary total_mapping reads $\{1:0, 3:1, 2:2, 0:3\}$.

This is this mapping that we are trying to undo. We want to return the list $[3,0,2,1]$, which we get by seeing that the $3$rd element in the our final ordering is $r_0$, the $0$th element is $r_1$, the $2$nd element is $r_2$, and the $1$st element is $r_3$. We plug this into __np.transpose__ like

$$\text{np.transpose}(\text{tensor}, \text{axes}=([3,0,2,1])),$$

which means that $3$rd axis will be mapped to the $0$th one, the $0$th axis will be mapped to the $1$st, the $2$nd will be mapped to the $2$nd, and the $1$st will be mapped to the $3$rd, which puts the axes in the same order as $\frac{\partial}{\partial A_{r_0 r_1 r_2 r_3}}$.  The line of code with does this is 



    undo_map_list = [total_mapping[i] for i in range(len(total_mapping))]

    return undo_map_list

