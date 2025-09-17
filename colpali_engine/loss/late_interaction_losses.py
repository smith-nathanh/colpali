import torch
import torch.nn.functional as F  # noqa: N812
from torch.nn import CrossEntropyLoss


class ColbertLoss(torch.nn.Module):
    def __init__(self, temperature: float = 0.02, normalize_scores: bool = True):
        """
        InfoNCE loss generalized for late interaction models.
        Args:
            temperature: The temperature to use for the loss (`new_scores = scores / temperature`).
            normalize_scores: Whether to normalize the scores by the lengths of the query embeddings.
        """
        super().__init__()
        self.ce_loss = CrossEntropyLoss()
        self.temperature = temperature
        self.normalize_scores = normalize_scores

    def forward(self, query_embeddings, doc_embeddings, neg_doc_embeddings=None):
        """
        query_embeddings: (batch_size, num_query_tokens, dim)
        doc_embeddings: (batch_size, num_doc_tokens, dim)
        neg_doc_embeddings: (batch_size, num_neg_doc_tokens, dim), optional
        """
        # Compute in-batch scores (always computed)
        scores = torch.einsum("bnd,csd->bcns", query_embeddings, doc_embeddings).max(dim=3)[0].sum(dim=2)

        if self.normalize_scores:
            query_lengths = (query_embeddings[:, :, 0] != 0).sum(dim=1).unsqueeze(-1)
            scores = scores / (query_lengths + 1e-8)

        if neg_doc_embeddings is not None:
            # Compute positive scores (diagonal)
            pos_scores = scores.diagonal().unsqueeze(1)  # (batch_size, 1)

            # Compute negative scores
            neg_scores = torch.einsum("bnd,bsd->bns", query_embeddings, neg_doc_embeddings).max(dim=2)[0].sum(dim=1)
            if self.normalize_scores:
                neg_scores = neg_scores / (query_lengths.squeeze(-1) + 1e-8)
            neg_scores = neg_scores.unsqueeze(1)  # (batch_size, 1)

            # Concatenate positive and negative scores
            all_scores = torch.cat([pos_scores, neg_scores], dim=1)  # (batch_size, 2)
            targets = torch.zeros(all_scores.shape[0], dtype=torch.long, device=all_scores.device)

            return self.ce_loss(all_scores / self.temperature, targets)
        else:
            # Original behavior: in-batch negatives only
            return self.ce_loss(scores / self.temperature, torch.arange(scores.shape[0], device=scores.device))


class ColbertNegativeCELoss(torch.nn.Module):
    def __init__(self, temperature: float = 0.02, normalize_scores: bool = True, in_batch_term=False):
        """
        InfoNCE loss generalized for late interaction models with negatives.
        Args:
            temperature: The temperature to use for the loss (`new_scores = scores / temperature`).
            normalize_scores: Whether to normalize the scores by the lengths of the query embeddings.
            in_batch_term: Whether to include the in-batch term in the loss.
        """
        super().__init__()
        self.ce_loss = CrossEntropyLoss()
        self.temperature = temperature
        self.normalize_scores = normalize_scores
        self.in_batch_term = in_batch_term

    def forward(self, query_embeddings, doc_embeddings, neg_doc_embeddings=None):
        """
        query_embeddings: (batch_size, num_query_tokens, dim)
        doc_embeddings: (batch_size, num_doc_tokens, dim)
        neg_doc_embeddings: (batch_size, num_neg_doc_tokens, dim), optional
        """

        if neg_doc_embeddings is not None:
            # Compute the ColBERT scores
            pos_scores = torch.einsum("bnd,bsd->bns", query_embeddings, doc_embeddings).max(dim=2)[0].sum(dim=1)
            neg_scores = torch.einsum("bnd,bsd->bns", query_embeddings, neg_doc_embeddings).max(dim=2)[0].sum(dim=1)

            loss = F.softplus(neg_scores / self.temperature - pos_scores / self.temperature).mean()

            if self.in_batch_term:
                scores = torch.einsum("bnd,csd->bcns", query_embeddings, doc_embeddings).max(dim=3)[0].sum(dim=2)
                if self.normalize_scores:
                    # find lengths of non-zero query embeddings
                    # divide scores by the lengths of the query embeddings
                    query_lengths = (query_embeddings[:, :, 0] != 0).sum(dim=1).unsqueeze(-1)
                    scores = scores / (query_lengths + 1e-8)
                loss += self.ce_loss(scores / self.temperature, torch.arange(scores.shape[0], device=scores.device))

            return loss / 2
        else:
            # When no negatives are provided (e.g., during evaluation), compute only in-batch loss
            scores = torch.einsum("bnd,csd->bcns", query_embeddings, doc_embeddings).max(dim=3)[0].sum(dim=2)
            if self.normalize_scores:
                # find lengths of non-zero query embeddings
                # divide scores by the lengths of the query embeddings
                query_lengths = (query_embeddings[:, :, 0] != 0).sum(dim=1).unsqueeze(-1)
                scores = scores / (query_lengths + 1e-8)
            return self.ce_loss(scores / self.temperature, torch.arange(scores.shape[0], device=scores.device))


class ColbertPairwiseCELoss(torch.nn.Module):
    def __init__(self):
        """
        Pairwise loss for ColBERT.
        """
        super().__init__()
        self.ce_loss = CrossEntropyLoss()

    def forward(self, query_embeddings, doc_embeddings):
        """
        query_embeddings: (batch_size, num_query_tokens, dim)
        doc_embeddings: (batch_size, num_doc_tokens, dim)

        Positive scores are the diagonal of the scores matrix.
        """

        # Compute the ColBERT scores
        scores = (
            torch.einsum("bnd,csd->bcns", query_embeddings, doc_embeddings).max(dim=3)[0].sum(dim=2)
        )  # (batch_size, batch_size)

        # Debug: Check embeddings and scores
        debug = torch.rand(1).item() < 0.05  # 5% chance to debug
        if debug:
            print(f"Loss debug - query_shape: {query_embeddings.shape}, doc_shape: {doc_embeddings.shape}")
            print(f"  query_mean: {query_embeddings.mean():.6f}, doc_mean: {doc_embeddings.mean():.6f}")
            print(f"  scores_shape: {scores.shape}, scores_mean: {scores.mean():.6f}")
            print(f"  scores_min: {scores.min():.6f}, scores_max: {scores.max():.6f}")

        # Positive scores are the diagonal of the scores matrix.
        pos_scores = scores.diagonal()  # (batch_size,)

        # Negative score for a given query is the maximum of the scores against all all other pages.
        # NOTE: We exclude the diagonal by setting it to a very low value: since we know the maximum score is 1,
        # we can subtract 1 from the diagonal to exclude it from the maximum operation.
        neg_scores = scores - torch.eye(scores.shape[0], device=scores.device) * 1e6  # (batch_size, batch_size)
        neg_scores = neg_scores.max(dim=1)[0]  # (batch_size,)

        # Debug: Check loss components
        if debug:
            print(f"  pos_scores: {pos_scores.mean():.6f}, neg_scores: {neg_scores.mean():.6f}")
            print(f"  pos-neg diff: {(pos_scores - neg_scores).mean():.6f}")

        # Compute the loss
        # The loss is computed as the negative log of the softmax of the positive scores
        # relative to the negative scores.
        # This can be simplified to log-sum-exp of negative scores minus the positive score
        # for numerical stability.
        # torch.vstack((pos_scores, neg_scores)).T.softmax(1)[:, 0].log()*(-1)
        loss = F.softplus(neg_scores - pos_scores).mean()

        # Debug: Check final loss
        if debug:
            print(f"  final_loss: {loss.item():.6f}")

        return loss


class ColbertPairwiseNegativeCELoss(torch.nn.Module):
    def __init__(self, in_batch_term=False):
        """
        Pairwise loss for ColBERT with negatives.
        Args:
            in_batch_term: Whether to include the in-batch term in the loss.
        """
        super().__init__()
        self.ce_loss = CrossEntropyLoss()
        self.in_batch_term = in_batch_term

    def forward(self, query_embeddings, doc_embeddings, neg_doc_embeddings):
        """
        query_embeddings: (batch_size, num_query_tokens, dim)
        doc_embeddings: (batch_size, num_doc_tokens, dim)
        neg_doc_embeddings: (batch_size, num_neg_doc_tokens, dim)
        """

        # Compute the ColBERT scores
        pos_scores = torch.einsum("bnd,bsd->bns", query_embeddings, doc_embeddings).max(dim=2)[0].sum(dim=1)
        neg_scores = torch.einsum("bnd,bsd->bns", query_embeddings, neg_doc_embeddings).max(dim=2)[0].sum(dim=1)

        loss = F.softplus(neg_scores - pos_scores).mean()

        if self.in_batch_term:
            scores = (
                torch.einsum("bnd,csd->bcns", query_embeddings, doc_embeddings).max(dim=3)[0].sum(dim=2)
            )  # (batch_size, batch_size)

            # Positive scores are the diagonal of the scores matrix.
            pos_scores = scores.diagonal()  # (batch_size,)
            neg_scores = scores - torch.eye(scores.shape[0], device=scores.device) * 1e6  # (batch_size, batch_size)
            neg_scores = neg_scores.max(dim=1)[0]  # (batch_size,)

            loss += F.softplus(neg_scores - pos_scores).mean()

        return loss / 2
