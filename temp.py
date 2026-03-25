from torchmetrics.multimodal import CLIPImageQualityAssessment

img_tensor = torch.rand(1, 3, 224, 224)  # Example image tensor
metric = CLIPImageQualityAssessment(prompts=("quality",))
score = metric(img_tensor)