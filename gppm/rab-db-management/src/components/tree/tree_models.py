"""
 ***********************************************************************************************************************
 * Creator     : changhi.lee
 * Create Date : 2026.09.17
 * Description : Confluence 페이지 트리를 표현하는 TreeNode / TreeStore 데이터 모델
 *               page의 계층 경로, 수정 시각, 본문 유무, labels 등 metadata를 보관하고
 *               트리 순회, 조회, snapshot 직렬화에 필요한 연산을 제공한다
 ***********************************************************************************************************************
 *                                                   Change History
 *********** ************ ************ ***************************************************** ************* *************
 * Change No  Change Date  Change User               Change Remark                               CR No       CTS Req No
 *********** ************ ************ ***************************************************** ************* *************
 *    1       2026.09.17   changhi.lee                  최초 작성
 *********** ************ ************ ***************************************************** ************* *************
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Optional

@dataclass
class TreeNode:
    """
    Confluence page 하나를 표현하는 tree node 모델.

    이 객체는 ConfluenceTreeDiscovery 단계에서 생성되며,
    tree snapshot 저장, 변경 비교, downloader/processor metadata patch에 사용된다.

    필드 구성:
        - page_id / title:
            Confluence page의 고유 id와 제목.
        - parent_id / children_ids:
            저장 가능한 형태의 parent-child 관계.
        - parent / children:
            런타임 탐색용 객체 참조. JSON 저장 대상에서는 제외된다.
        - ancestor_ids / ancestor_titles / depth:
            page의 계층 경로 정보.
        - last_modified_date:
            updated 여부 판단에 사용하는 수정 시각.
        - has_body / has_children / is_container:
            본문이 있는 문서인지, 구조용 page인지 판단하는 metadata.
        - page_tree / page_url / labels:
            dump, embedding, DB metadata로 downstream에 전달되는 정보.
    """
    page_id: str
    title: str = ""
    parent_id: Optional[str] = None
    children_ids: list[str] = field(default_factory=list)

    ancestor_ids: list[str] = field(default_factory=list)
    ancestor_titles: list[str] = field(default_factory=list)
    depth: Optional[int] = None

    last_modified_date: Optional[str] = None

    has_body: Optional[bool] = None
    has_children: Optional[bool] = None
    is_container: bool = False
    is_bottom_parent: bool = False

    page_tree: Optional[str] = None
    page_url: Optional[str] = None
    page_url_tiny: Optional[str] = None
    labels: list[str] = field(default_factory=list)

    parent: Optional["TreeNode"] = field(default=None, repr=False)
    children: list["TreeNode"] = field(default_factory=list, repr=False)

    
    def add_child(self, child: "TreeNode") -> None:
        """
        현재 node에 child node를 연결한다.

        처리 내용:
            1. children_ids에 child.page_id를 추가한다.
            2. children 객체 목록에 child를 추가한다.
            3. child.parent와 child.parent_id를 현재 node 기준으로 갱신한다.

        주의:
            children_ids는 중복을 막지만, children 객체 목록은 중복 체크를 하지 않는다.
            일반적으로 TreeStore.build_links()로 전체 링크를 재구성하는 흐름을 더 많이 사용한다.
        """
        if child.page_id not in self.children_ids:
            self.children_ids.append(child.page_id)
        self.children.append(child)
        child.parent = self
        child.parent_id = self.page_id

    
    def is_leaf(self) -> bool:
        """
        현재 node가 leaf page인지 판단한다.

        판단 기준:
            - children_ids가 있으면 leaf가 아니다.
            - has_children=True이면 children_ids가 비어 있어도 leaf가 아니다.
            - 둘 다 아니면 leaf로 본다.

        has_children을 함께 보는 이유:
            일부 상황에서는 children_ids가 아직 로드되지 않았더라도
            Confluence metadata상 child가 존재한다는 사실을 알 수 있기 때문이다.
        """
        if self.children_ids:
            return False
        if self.has_children is True:
            return False
        return True

    
    def walk_descendants(self) -> Iterator["TreeNode"]:
        """
        현재 node 아래의 모든 descendant node를 DFS 순서로 순회한다.

        Returns:
            Iterator[TreeNode]:
                child, grandchild, ... 순서의 descendant iterator.
        """
        for child in self.children:
            yield child
            yield from child.walk_descendants()

    
    def to_dict(self) -> Dict[str, Any]:
        """
        TreeNode를 JSON 저장 가능한 dict로 변환한다.

        parent / children 객체 참조는 저장하지 않는다.
        해당 필드는 순환 참조를 만들 수 있고, 저장 후 TreeStore.build_links()로 다시 복원할 수 있다.

        Returns:
            Dict[str, Any]:
                tree snapshot JSON에 저장될 node metadata.
        """
        return {
            "page_id": self.page_id,
            "title": self.title,
            "parent_id": self.parent_id,
            "children_ids": list(self.children_ids),
            "ancestor_ids": list(self.ancestor_ids),
            "ancestor_titles": list(self.ancestor_titles),
            "depth": self.depth,
            "last_modified_date": self.last_modified_date,
            "has_body": self.has_body,
            "has_children": self.has_children,
            "is_container": self.is_container,
            "is_bottom_parent": self.is_bottom_parent,
            "page_tree": self.page_tree,
            "page_url": self.page_url,
            "page_url_tiny": self.page_url_tiny,
            "labels": list(self.labels),
        }

    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TreeNode":
        """
        JSON에서 읽은 dict를 TreeNode 객체로 복원한다.

        parent / children 참조는 여기서 복원하지 않는다.
        모든 node가 TreeStore에 추가된 뒤 TreeStore.build_links()에서 parent-child 객체 참조를 재구성한다.

        Args:
            data:
                TreeNode.to_dict()로 저장된 node metadata.

        Returns:
            TreeNode:
                복원된 node 객체.
        """
        return cls(
            page_id=str(data["page_id"]),
            title=data.get("title", ""),
            parent_id=data.get("parent_id"),
            children_ids=list(data.get("children_ids") or []),
            ancestor_ids=list(data.get("ancestor_ids") or []),
            ancestor_titles=list(data.get("ancestor_titles") or []),
            depth=data.get("depth"),
            last_modified_date=data.get("last_modified_date"),
            has_body=data.get("has_body"),
            has_children=data.get("has_children"),
            is_container=bool(data.get("is_container", False)),
            is_bottom_parent=bool(data.get("is_bottom_parent", False)),
            page_tree=data.get("page_tree"),
            page_url=data.get("page_url"),
            page_url_tiny=data.get("page_url_tiny"),
            labels=list(data.get("labels") or []),
        )

class TreeStore:
    """
    TreeNode들을 page_id 기준으로 관리하는 tree container.

    TreeStore는 JSON 저장/로드 가능한 구조와 런타임 탐색 가능한 구조를 함께 제공한다.

    내부 구조:
        - nodes_by_id:
            page_id -> TreeNode 매핑.
        - root_ids:
            parent_id가 없거나 parent가 tree 안에 없는 최상위 node id 목록.

    주요 역할:
        1. TreeNode 추가/조회
        2. parent_id / children_ids 기반 parent-child 객체 링크 생성
        3. bottom parent 표시
        4. subtree id 수집
        5. JSON 직렬화/역직렬화
    """

    def __init__(self) -> None:
        """
        빈 TreeStore를 초기화한다.
        """
        self.nodes_by_id: dict[str, TreeNode] = {}
        self.root_ids: list[str] = []

    def add_node(self, node: TreeNode) -> None:
        """
        TreeNode를 page_id 기준으로 저장한다.

        동일 page_id가 이미 있으면 새 node로 덮어쓴다.
        """
        self.nodes_by_id[node.page_id] = node

    def get(self, page_id: str) -> Optional[TreeNode]:
        """
        page_id에 해당하는 TreeNode를 반환한다.

        Args:
            page_id:
                조회할 Confluence page id.

        Returns:
            Optional[TreeNode]:
                존재하면 TreeNode, 없으면 None.
        """
        return self.nodes_by_id.get(str(page_id))

    def __contains__(self, page_id: str) -> bool:
        """
        page_id가 TreeStore에 존재하는지 확인한다.

        Example:
            if page_id in tree:
                ...
        """
        return str(page_id) in self.nodes_by_id

    def __len__(self) -> int:
        """
        TreeStore에 저장된 node 개수를 반환한다.
        """
        return len(self.nodes_by_id)

    def iter_nodes(self):
        """
        TreeStore에 저장된 모든 TreeNode를 순회한다.

        Returns:
            Iterable[TreeNode]:
                nodes_by_id.values().
        """
        return self.nodes_by_id.values()

    def build_links(self) -> None:
        """
        parent_id와 children_ids 정보를 기반으로 런타임 parent-child 객체 참조를 재구성한다.

        처리 흐름:
            1. 모든 node의 parent / children 참조를 초기화한다.
            2. node.parent_id가 TreeStore 안에 있으면 parent.children에 node를 연결한다.
            3. parent가 없거나 parent_id가 TreeStore 안에 없으면 root_ids에 넣는다.

        사용 시점:
            - JSON에서 tree를 로드한 직후
            - subtree 삭제/이동/병합 후
            - 새 node를 대량 추가한 후

        주의:
            children_ids는 저장용 관계 정보이고,
            children은 탐색용 객체 참조다.
        """
        self.root_ids = []

        for node in self.nodes_by_id.values():
            node.parent = None
            node.children = []

        for node in self.nodes_by_id.values():
            if node.parent_id and node.parent_id in self.nodes_by_id:
                parent = self.nodes_by_id[node.parent_id]
                parent.children.append(node)
                node.parent = parent
            else:
                self.root_ids.append(node.page_id)

    def mark_bottom_parents(self) -> None:
        """
        bottom parent 여부를 계산해 각 node.is_bottom_parent에 표시한다.

        bottom parent 정의:
            - child를 가지고 있고
            - 그 child 중 leaf가 하나라도 있는 parent

        사용 목적:
            ConfluenceTreeDiscovery에서 hidden 구조 변화를 탐지하기 위한 anchor로 사용된다.
            예를 들어 기존 leaf 아래에 새 child가 생기는 경우 direct parent 비교만으로 놓칠 수 있으므로,
            bottom parent 주변을 ancestor CQL로 다시 확인한다.
        """
        for node in self.nodes_by_id.values():
            if not node.children:
                node.is_bottom_parent = False
                continue

            # 자식 중 leaf가 하나라도 있으면 bottom parent로 간주한다.
            node.is_bottom_parent = any(child.is_leaf() for child in node.children)

    def get_subtree_ids(self, root_id: str) -> list[str]:
        """
        root_id를 기준으로 subtree에 포함된 모든 page_id를 반환한다.

        반환 목록에는 root_id 자신도 포함된다.

        Args:
            root_id:
                subtree root page id.

        Returns:
            list[str]:
                root와 모든 descendant page_id 목록.
        """
        root = self.get(root_id)
        if not root:
            return []
        result = [root.page_id]
        result.extend(n.page_id for n in root.walk_descendants())
        return result

    def to_dict(self, metadata: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """
        TreeStore 전체를 JSON 저장 가능한 dict로 변환한다.

        저장 내용:
            - metadata:
                schema_version, node_count, saved_at 등 부가 정보
            - nodes:
                page_id -> TreeNode.to_dict()
            - root_ids:
                root node id 목록

        Args:
            metadata:
                저장 파일에 함께 포함할 메타데이터.
                없으면 기본 metadata만 생성한다.

        Returns:
            dict[str, Any]:
                tree snapshot JSON payload.
        """
        meta = dict(metadata or {})
        meta.setdefault("schema_version", "1.0")
        meta.setdefault("node_count", len(self.nodes_by_id))
        meta["nodes"] = {
            page_id: node.to_dict()
            for page_id, node in self.nodes_by_id.items()
        }
        meta["root_ids"] = list(self.root_ids)
        return meta

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TreeStore":
        """
        JSON payload를 TreeStore 객체로 복원한다.

        처리 흐름:
            1. payload["nodes"]에서 TreeNode를 복원한다.
            2. 모든 node를 TreeStore에 추가한다.
            3. build_links()로 parent-child 객체 참조를 복원한다.
            4. mark_bottom_parents()로 bottom parent 정보를 다시 계산한다.

        Args:
            payload:
                TreeStore.to_dict()로 저장된 JSON payload.

        Returns:
            TreeStore:
                복원된 tree 객체.
        """
        tree = cls()
        nodes = payload.get("nodes", {}) or {}

        for _, node_data in nodes.items():
            node = TreeNode.from_dict(node_data)
            tree.add_node(node)

        tree.build_links()
        tree.mark_bottom_parents()
        return tree