import numpy as np

class BinarySumTree():
    """
    A binary sum tree
    """
    def __init__(self, num_leaves):
        """
        Creates a binary sum tree with the given number of leaves

        num_leaves : The number of leaves in the tree
        """
        self.num_leaves = num_leaves
        self.size = 0
        self.current_index = 0

        self.tree = np.zeros(2 * self.num_leaves - 1)

    def __len__(self):
        """
        Returns the number of values added up to the number of leaves in the
        tree
        """
        return self.size

    def _update_parents(self, index, value):
        """
        Updates all the parent nodes going up to the root to accomodate the
        addition of a new node

        index : The index of the new node
        value : The value of the new node
        """
        change_in_value = value - self.tree[index]
        parent = index

        # Keep updating until the root node is reached
        while(parent != 0):
            parent = (parent - 1) // 2
            self.tree[parent] += change_in_value

    def _leaf_start_index(self):
        """
        Returns the starting index of the leaves
        """
        return self.num_leaves - 1

    def _leaf_idx_to_real(self, leaf_index):
        """
        Converts the index of a leaf relative to the other leaves to the index
        in the tree (num_leaves - 1 + leaf_index)

        leaf_index : The index of the leaf to be convert to the tree index
        """
        return self._leaf_start_index() + leaf_index

    def add(self, value):
        """
        Pushes the given value onto the sum tree. When the tree is at capacity,
        the values will be replaced starting with the first one.

        value: The value of the item
        """
        self.set(value, self.current_index)
        self.current_index = (self.current_index + 1) % self.num_leaves

        if(self.size < self.num_leaves):
            self.size += 1

    def set(self, value, index):
        """
        Sets the value of the leaf at the leaf index given

        value : The value of the leaf
        index : The index of the leaf
        """
        tree_index = self._leaf_idx_to_real(index)
        self._update_parents(tree_index, value)
        self.tree[tree_index] = value

    def get(self, index):
        """
        Retrieves the node at the given index

        index : The index of the node to retrieve
        """
        return self.tree[index]

    def get_leaf(self, leaf_index):
        """
        Returns the leaf with the given index relative to the leaves

        leaf_index : The index of the leaf relative to other leaves
        """
        tree_index = self._leaf_idx_to_real(leaf_index)
        return self.get(tree_index)

    def get_leaves(self):
        """
        Returns all the added leaves in the tree
        """
        leaf_start = self._leaf_start_index()
        return self.tree[leaf_start:leaf_start + self.size]

    def sum(self):
        """
        Returns the sum of the tree (the value of the root)
        """
        return self.tree[0]

    def next_index(self):
        """
        Returns the leaf index of the next value added
        """
        return self.current_index
