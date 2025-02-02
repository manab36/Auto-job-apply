from Linkedin.Linkedin import Linkedin
from Linkedin.LinkedinJobApply import LinkedinJobApply
from Linkedin.LinkedinPostScrape import LinkedinPostScrape

"""
source auto_apply_evn/bin/activate
watch -n 1 nvidia-smi
"""


# # Linkedin()
# LinkedinJobApply_obj= LinkedinJobApply()
# LinkedinJobApply_obj.run()

LinkedinJobApply_obj= LinkedinPostScrape()
LinkedinJobApply_obj.run()


