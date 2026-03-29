class CommodityRecord:
    """
    Entidade de domínio (Data Object). 
    Armazena o par chave-valor para indexação e busca.
    """
    def __init__(self, key, data):
        self.key = key  # Chave de indexação (Primary Key)
        self.data = data # Payload (Dicionário com atributos do CSV)

class Node:
    """
    Abstração de um nó em uma árvore binária.
    Contém ponteiros para subárvores e metadados para balanceamento.
    """
    def __init__(self, record):
        self.record = record
        self.left = None
        self.right = None
        self.height = 1 # Necessário para o cálculo do fator de balanceamento na AVL

class BST:
    """
    Binary Search Tree (BST) não balanceada.
    Operações de busca/inserção possuem complexidade média O(log n).
    """
    def __init__(self):
        self.root = None

    def insert(self, record):
        """
        Inserção Iterativa para evitar Stack Overflow em árvores degeneradas.
        Garante que a propriedade da BST (Esq < Raiz < Dir) seja mantida.
        """
        if not self.root:
            self.root = Node(record)
            return

        curr = self.root
        while True:
            if record.key < curr.record.key:
                if curr.left is None:
                    curr.left = Node(record)
                    break
                curr = curr.left
            elif record.key > curr.record.key:
                if curr.right is None:
                    curr.right = Node(record)
                    break
                curr = curr.right
            else:
                break # Tratamento de chaves duplicadas (idempotência)

    def get_height_iterative(self):
        """
        Cálculo de altura utilizando DFS (Depth-First Search) iterativo com pilha.
        Evita os limites de recursão do Python em datasets de larga escala.
        """
        if not self.root:
            return 0
        max_depth = 0
        stack = [(self.root, 1)]
        while stack:
            node, depth = stack.pop()
            max_depth = max(max_depth, depth)
            if node.left:
                stack.append((node.left, depth + 1))
            if node.right:
                stack.append((node.right, depth + 1))
        return max_depth

    def search(self, key):
        """
        Busca binária padrão. Complexidade O(h), onde h é a altura.
        No pior caso (árvore lista), torna-se O(n).
        """
        curr = self.root
        while curr:
            if key == curr.record.key:
                return curr.record
            elif key < curr.record.key:
                curr = curr.left
            else:
                curr = curr.right
        return None

    def search_by_attribute(self, attribute_name, value):
        """
        Busca por atributo não indexado. Requer percurso completo (Linear Search).
        Complexidade O(n), independente da estrutura da árvore.
        """
        results = []
        if not self.root:
            return results
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.record.data.get(attribute_name) == value:
                results.append(node.record)
            # Exploração de ambos os ramos, pois o atributo não dita a ordem
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)
        return results

class AVLTree:
    """
    Árvore Binária de Busca Auto-Balanceada (Adelson-Velsky e Landis).
    Mantém a invariante |altura(esq) - altura(dir)| <= 1.
    Garantiu complexidade O(log n) para busca, inserção e remoção.
    """
    def __init__(self):
        self.root = None

    def get_height(self, node):
        """Acesso seguro ao atributo height (trata referências None)."""
        return node.height if node else 0

    def get_balance(self, node):
        """Calcula o Fator de Balanceamento (FB)."""
        return self.get_height(node.left) - self.get_height(node.right) if node else 0

    def right_rotate(self, y):
        """
        Rotação Simples à Direita (LL). 
        Reestabelece o equilíbrio quando a subárvore à esquerda está mais pesada.
        """
        x = y.left
        T2 = x.right
        x.right = y
        y.left = T2
        # Atualização de metadados de altura após reestruturação topológica
        y.height = 1 + max(self.get_height(y.left), self.get_height(y.right))
        x.height = 1 + max(self.get_height(x.left), self.get_height(x.right))
        return x

    def left_rotate(self, x):
        """
        Rotação Simples à Esquerda (RR).
        Utilizada quando a inserção ocorre na subárvore direita do filho direito.
        """
        y = x.right
        T2 = y.left
        y.left = x
        x.right = T2
        x.height = 1 + max(self.get_height(x.left), self.get_height(x.right))
        y.height = 1 + max(self.get_height(y.left), self.get_height(y.right))
        return y

    def insert(self, record):
        """Interface pública para inserção recursiva."""
        self.root = self._insert(self.root, record)

    def _insert(self, root, record):
        """
        Inserção recursiva com etapa de 'backtracking' para rebalanceamento.
        """
        if not root:
            return Node(record)

        if record.key < root.record.key:
            root.left = self._insert(root.left, record)
        elif record.key > root.record.key:
            root.right = self._insert(root.right, record)
        else:
            return root # Chaves únicas

        # Atualização da altura no retorno da recursão
        root.height = 1 + max(self.get_height(root.left), self.get_height(root.right))

        # Verificação da violação da invariante da AVL
        balance = self.get_balance(root)

        # Casos de Desbalanceamento e Rotações:
        
        # Caso Left-Left (LL): Rotação simples à direita
        if balance > 1 and record.key < root.left.record.key:
            return self.right_rotate(root)
        
        # Caso Right-Right (RR): Rotação simples à esquerda
        if balance < -1 and record.key > root.right.record.key:
            return self.left_rotate(root)

        # Caso Left-Right (LR): Rotação dupla (esquerda-direita)
        if balance > 1 and record.key > root.left.record.key:
            root.left = self.left_rotate(root.left)
            return self.right_rotate(root)
        
        # Caso Right-Left (RL): Rotação dupla (direita-esquerda)
        if balance < -1 and record.key < root.right.record.key:
            root.right = self.right_rotate(root.right)
            return self.left_rotate(root)

        return root

    def search(self, key):
        """Busca O(log n) garantida pela topologia balanceada."""
        curr = self.root
        while curr:
            if key == curr.record.key:
                return curr.record
            elif key < curr.record.key:
                curr = curr.left
            else:
                curr = curr.right
        return None

    def search_by_attribute(self, attribute_name, value):
        """
        Busca exaustiva. Mesmo em uma AVL, a ausência de índice no atributo 
        força complexidade O(n).
        """
        results = []
        if not self.root:
            return results
        stack = [self.root]
        while stack:
            node = stack