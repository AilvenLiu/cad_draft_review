Project Proposal: CAD Draft Review for Petroleum Industry

0. the whole propose is understand the CAD draft from the Petroleum area and review it to find out any mistakes (based on the rules) and somewhere could be optimized (optional requirement, based on the rule-based knowledge base).
1. the CAD drafts consist of signs and lines, in which the signs is the ulimited (which means, although the common and usual signs are limited in its number of category, we always encounter the new kind of sign in the draft, which delivers a challenge of incremental learning or consecutive learning or object detection in open world and so on)
2. there're one or more lines link each sign to others.
2. the source data consist of 
2.1 lots of CAD image. in which some are totally labeled following the common object detect format(YOLO or VOC or COCO, etc.), some are partially labeled (parts of the signs are labeled while others are not labeled), and some are totally not labled.
2.2 the sign table. in which the shape of sign and its label(name) are corresponded, where the shape of sign are strictly same between table and image
2.3 code in image. Some signs are indentical in their shape while differentiate by the code(text) near by them in image. For example the Pump are mostly identical in shape, but different kind of them are differentiate by the near code(text) such as "P-20058", "P-10223".
2.4 the rule files. It is the excel table or just text lines, but is not formatable. Each line in file delivers a rule to describe a design demand of the CAD project for example there's must a Protector after the specified Pump (differentiate by the near code). Addentially, the rules are expected to be integrated into the knowledge base. 
3. there's no labels for lines, but the lines have only 2 main categories: dotted line and solid line, to indicate different manners of connection between signs.
4. the CAD image may very huge in shape (may be large than 10000 by 10000 pixels) and provided in PDF format, which far exceed the ability of normal CNN based OD model (Yolo as example).

Addition:
1. use 3rd-part module fitz to convert pdf into png in the project 
2. it should be emphasized of the open world ability (or the same ability to handle the unseen catagory of CAD signs)
3. the final propose are expected to be a system with tensorRT or other inference framework based (but be unnecessary to consider to convert model to inference model at that moment).
4. the most improtant thing is the innovativeness: it is expected to propose some effective and noval (original) design.