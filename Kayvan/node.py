class Node:
    def __init__(self, attribute, value, left_node, right_node, leaf):
        self.attribute = attribute
        self.value = value
        self.left_node = left_node
        self.right_node = right_node
        self.leaf = leaf

    def print_tree(self, indent=""):
        if self.leaf is True:
            print(f"{indent}Classification: {self.value}")
        else:
            print(f"{indent}Attribute: {self.attribute}, Value: {self.value}")
            print(f"{indent}--> Left Branch:")
            self.left_node.print_tree(indent + "    ")
            print(f"{indent}--> Right Branch:")
            self.right_node.print_tree(indent + "    ")

    def prune_tree(self):
        if not self.leaf:
            self.left_node.prune_tree()
            self.right_node.prune_tree()

            if self.left_node.leaf and self.right_node.leaf:
                if self.left_node.value == self.right_node.value:
                    self.attribute = -1
                    self.value = self.left_node.value
                    self.left_node = None
                    self.right_node = None
                    self.leaf = True
                    print(f"Pruned node")

    def traverse_tree(self,instance):
        if self.leaf:
            return self.value
        if(instance[self.attribute] < self.value):
            return self.left_node.traverse_tree(instance)
        else:
            return self.right_node.traverse_tree(instance)