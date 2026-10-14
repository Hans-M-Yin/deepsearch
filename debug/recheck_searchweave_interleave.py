"""Recheck all 81 previously filtered interleave cases, without new inference."""
import csv
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path('opensearch_infer/analysis/searchweave_v2_t2i_review')
OUT = ROOT / 'interleave_recheck'

# Evidence group, image bridge, explanation. An observed answer is not a proof
# that the required visual attribute can be retrieved in text.
NOTES = {
 'q_000015': ('text_evidence', 'Operation Santa Fe→省份→CSCL Star船首', '问题直接给船名CSCL STAR。消融文本搜索turn12船舶资料已给Chinese name中海之星，省份及战争图片均不再必需。'),
 'q_000034': ('answer_only', 'Colossi of Memnon→Amenhotep III头雕断冠', 'GPT回答冠顶缺损，但没有无图输入对照或文字描述证据。完整工具运行确实搜图观察目标头雕；可能是记忆了著名展品，不能据一次答对认定视觉冗余。'),
 'q_001393': ('text_evidence', 'In Rock头像→Ian Gillan→Jesus Christ Superstar封面编号', '消融turn25发行资料正文直接写DXSA 7206。模型最终选错A Rock Opera，关键答案已在文字中；无需对封面左上角做OCR。'),
 'q_003086': ('unverified_bypass', 'Trevi Fountain画作铭文→Clement XII→Neri Corsini', '铭文和人物家族关系可能由喷泉文字介绍取得，但所有保存运行都错到不同年代Neri。没有成功文字链，原先过滤只是可绕过假设，需暂缓。'),
 'q_004785': ('input_answer', '已给书封面→同一本书展开书衣', '实际查看GPT输入图，红马肚子下的城市建筑已经清楚可见。无需另搜展开书衣，但仍需看给定图片，不能归为完全不需要视觉。'),
 'q_005918': ('question_shortcut', 'Raphael墓→School of Athens→Aristotle→Classical Greece', '问题已经完整点名Second Athenian League与Delian League并询问财政差异。直接查两联盟制度即可，Aristotle所在壁画对末端答案无约束；GPT细节是否完整另需复核。'),
 'q_007046': ('judge_uncertain', 'Meteora最高修道院→藏骨室墙材料', 'GPT只答石砌墙，金标要求brick and stone。judge认为stone足以且推断传统建筑常混砖石。不能把未答出砖的宽松判对当作精确视觉线索已被绕过。'),
 'q_011890': ('text_evidence', '守门员球衣品牌→Portsmouth保杯年数', '消融turn27的1939 FA Cup历史文字给二战停赛后七年再办决赛。问题中1930年代决赛和长期持杯也能从足球史识别球队，不必从另一张球衣图观察品牌。'),
 'q_012124': ('judge_uncertain', 'Racławice画中鹰徽→1927设计来源', 'GPT答1924 Polish coin，金标是纪念Paderewski的medal；消融又答plaque。judge把三种物件概括成纪念金属物件后接受。应核对物件类别及gold，当前不能凭正确标签确认严格答对。'),
 'q_012207': ('answer_only', 'Carlsberg Elephant Gate圆形毯章字母', '文本找到象上儿童名字首字母T/V/H/P，相关read_url多次明确没有圆形medallion内字母描述。消融最后答C但无对应文字证据，不能证明特定圆章可文本回答。'),
 'q_012243': ('text_evidence', 'Oglethorpe→受托人接见画→Senauki', '消融turn7 Senauki生平直接写Tomochichi死后她与Toonahowi共同领导。人物及会议文字资料可替代画中粉裙人物；候选表中的文字证据支持绕过。'),
 'q_012502': ('answer_only', 'Puskás→1960夺冠图→守门员球衣', 'GPT仅答black；消融错找1986并答green。黑色是常见守门员球衣先验，一次命中既可能是记忆也可能猜测，不能证明正确比赛图片不必观察。'),
 'q_013632': ('unverified_bypass', 'Wahlburgers开幕厨师衣标→Ralph Lauren', 'GPT答1972/24色，是高度典型的Polo产品史答案，可能直接按末问识别品牌。但正确开幕厨师衣标未由文本验证，消融误走Lacoste；不能确认中间品牌图片可被文本闭合。'),
 'q_013742': ('text_evidence', '签字场景人物位置→Sutherland纠错', '完整工具轨迹只有t2t/read_url，turn1明确划去预印国家名、重写并签注。历史事件文字直接描述纠错者和行为，人物站位图片可绕过；GPT也已答核心过程但被判错。'),
 'q_020271': ('answer_only', 'Correggio Danaë→Klimt Danaë手指位置', '消融文字提parted lips，但read_url也明确没有fingers proximity描述。GPT和消融答lips只能证明无搜图命中；嘴唇是有限候选之一，未证明手指位置有文字依据。'),
 'q_020485': ('answer_only', 'Yankee Stadium→Macombs Dam Park航拍球场数', 'GPT答3，但消融错认Fair Park，完整工具有读图。数字可能来自知识或三场的常见布局先验，尚无正确公园的文字计数证据。'),
 'q_020546': ('unverified_bypass', 'Persephone→Proserpine→Jane Morris刺绣署名', '末端是商业署名史，理论上可用模特/公司文字资料替代画作，但现有文本轨迹没有商业成功那段正确证据。GPT概括营销原因获通过，不足以确认完整文字替代链。'),
 'q_020564': ('text_evidence', 'Hyde Park Corner拱门→Peel→误杀与法律规则', '消融turn2检索文字已描述Drummond误被当Peel枪击及M’Naghten规则。拱门提案图不能为受害者或法律规则提供独有信息。'),
 'q_020834': ('answer_only', '政治漫画Uncle Sam→WWI招募海报帽带', '蓝色帽带是著名Uncle Sam海报的典型属性，可能被模型记住。现有正确标签不等于找到一段描述帽带颜色的文字，也不证明模型对陌生视觉目标无需搜图。'),
 'q_021236': ('answer_only', '球员→赛事→冠军庆祝T恤文字', 'CHAMPIONS是冠军T恤极强先验。消融把队员/赛事走错也答同字，显示答案区分力低；支持改题或按无搜图规则过滤，不能宣称正确目标的字已由文本查到。'),
 'q_021734': ('answer_only', '奖杯公司AT&T→Bell电话照片→左右手', 'GPT答right但消融错识SK Telecom并答left。左右手二选一存在猜中风险；现有数据未提供正确照片手部的文本证据。'),
 'q_022009': ('answer_only', 'Matilda雕像→Vita Mathildis画中内袍', 'GPT答blue，消融没有有效判分。蓝色衣物属于常见绘画属性；需有对应手稿文字说明或图像对照后才能称图像可绕过。'),
 'q_022274': ('unverified_bypass', 'Foreign Correspondent→Phantom Lady海报顶头像→Franchot Tone', '演员表及Oscar制度史可能代替海报，但未建立正确演员文字链。GPT实际答新设Supporting Actor/Actress，judge因没重述原因拒绝，显示标签问题；这仍不证明桥接人物可以纯文本锁定。'),
 'q_022556': ('answer_only', '捕获舰Ville de Paris→Barfleur→战斗画中左臂', '无输入图的GPT命中pointing，消融却答用boat-hook救人。动作细节确实是目标画局部，可能记忆或常见手势猜中，当前没有文本说明左臂的证据。'),
 'q_022639': ('answer_only', 'Iowa县→Squirrel Cage Jail屋顶', 'weathervane/finial是屋顶顶端常见物，金标也接受两种。GPT命中不能排除建筑常识猜测，未拿到正确cupola顶部对象的文字描述。'),
 'q_022726': ('unverified_bypass', 'Vespucci→Waldseemüller地图铭文→Castile法律身份', '地图作者及殖民地法律史都可查文本，但GPT概括kingdoms of Castile、消融只答assigned to Castile，细节判分不同。需核对条约与法律转变，以及是否真正从文本识别铭文王国。'),
 'q_022740': ('question_shortcut', '飞机翼徽→德国陆军战术名称', '问题直接给fast-moving wedge和encirclement cauldron，是对Keil und Kessel的释义提示；可以检索战术描述或翻译关键术语，翼徽图片没有独有约束。消融最终只答Kessel，标签完整性也值得核对。'),
 'q_022847': ('answer_only', 'Wat Phra Kaew→Chakri Maha Prasat屋顶层数', '问题已给目标建筑名，绕过了前面关联，但层数3本身仍是视觉计数。GPT答3、消融答7，未找到文字计数，所以目前只证明无搜图答对，不证明视觉线索可文本获取。'),
 'q_022987': ('text_evidence', '城堡象征→Malta Order接见照片→Festing辞职争议', '消融turn3等新闻正文描述Myanmar避孕用品分发争议。末问Grand Chancellor/辞职/教廷争议也足以直接检索事件，不必以接见照片识别人。'),
 'q_022995': ('answer_only', 'Fender Jaguar旧广告→两音箱摆放', 'GPT答上下叠放符合常见head/cabinet结构，消融答side by side。没有正确广告的文字空间说明；这属于典型先验可能命中，不能认定文本已替代广告图。'),
 'q_023335': ('text_evidence', 'Tunnel of Love海报导演→Gene Kelly首部独导', '问题直接点名Gene Kelly；消融turn1其生平给Invitation to the Dance及商业失败，海报导演署名完全冗余。'),
 'q_023371': ('text_evidence', 'jubilee阳台右端女子→Windsor命名者', '消融turn17 House of Windsor正文直接给Stamfordham。所有符合王室成员背景的女子都导向相同王朝，阳台位置对命名者没有独有信息。'),
 'q_023533': ('question_shortcut', 'Book of Kells另页→Mary衣饰纹样', '问题已写decorative pattern of white diamonds，金标仍是white stars/diamond shapes。GPT保存的生成文本先从输入图识别Chi Rho/Book of Kells，再声称已有描述/复制品显示white diamonds并答lozenges，获判正确；不是纯文本随机猜测。初始图非Mary页，但题干已泄露核心形状，主动搜目标页不是获得此答案的必需步骤。'),
 'q_023543': ('unverified_bypass', 'Strasbourg战斗照片右侧舰→Provence炮去向', '军舰参战名单及打捞史可能替代舰影识别，但现有文字运行混入Normandie/其他船，未取得正确炮去向证据。GPT无搜图答对可以作为筛选信号，不能称文字链已验证。'),
 'q_023760': ('answer_only', 'Last Supper→Magdalene→La Tour腰带', '消融文字有red skirt，却多次明确不描述sash，最终答red不能说明取得腰带文字证据。需要区分红裙与红腰带，避免judge或模型把相邻部件颜色混同。'),
 'q_023905': ('unverified_bypass', 'Phantom Lady海报中头像→Ella Raines替演原因', '女演员表与替演史可能可查，但GPT答另一部影片及另一演员，文本消融也错找Coleen Gray。没有正确替演事件的文字证据，应从确定过滤降为待核。'),
 'q_024662': ('text_evidence', '单曲图蓝发女子→Alissa White-Gluz祖父母经历', '消融turn12生平正文直接连到First Day in Hell；歌词灵感和歌手身份可从专辑及成员文字取得，单曲封面不提供独有信息。'),
 'q_025108': ('text_evidence', '古希腊瓶画火炬者→Hecate三身雕像史', '消融turn9正文直接给Alcamenes和Temple of Wingless Nike；问题里火炬/三身也是Hecate典型图像学，瓶画不是末端历史知识的唯一通道。'),
 'q_025139': ('unverified_bypass', 'Phelps Stokes→University Settlement外墙刻字', '建筑名可能等于刻字，但名字不能逻辑保证实际lintel文字，可能是旧名/捐赠铭文。GPT答NEIGHBORHOOD GUILD，消融没有正确刻字证据，应暂缓过滤。'),
 'q_025390': ('question_shortcut', '桥施工图→Manhattan→mainland社区', '哪一个通常属于Manhattan却在美国大陆上的社区，是可直接查的Marble Hill地理特征。消融错误认Brooklyn Bridge仍答Marble Hill且文本搜索给mainland属性，说明桥施工照片不是独有线索。'),
 'q_025468': ('quality_problem', '红衫Carrefour→门店第二桶价格', '价格确实需要观察具体图，但门店/日期/视频身份未限定，无法稳定找到唯一目标。应因目标不可定位过滤或改写，不能称无需图片。'),
 'q_025487': ('answer_only', 'Amsterdam Olympic Stadium→1928开幕式服色', 'white是强服装先验；消融错走Chelsea/1948也答white获通过。不能证明正确1928代表团颜色在文本里，更不能把错误场景命中当搜图可替代证据。'),
 'q_025769': ('text_evidence', '五名神风队员抱狗照片→Yukio Araki遗书', '问题已经详细描述著名照片；消融turn1遗书文字给loyalty to Emperor/devotion to parents。图注及遗书可用文本搜索获取，无须先看目标照片。'),
 'q_026394': ('answer_only', 'Grand Rapids→Meyer May House砖墩顶物', '问题直接给Meyer May House但planter/urn仍是具体建筑细节。消融只取到electrified lanterns并答灯，GPT答urn，尚无正确砖墩对象文字证据。'),
 'q_026444': ('answer_only', 'Kitten Kong→1933 King Kong右侧飞机', '消融读取Peter Jackson 2005版本及generic plane brought down，没获取1933目标帧的最右飞机状态。常见战斗结局能猜中crashing，但未证明局部视觉线索可文本回答。'),
 'q_026458': ('text_evidence', 'Declaration of Independence画中接收者→Dunlap印本', '首版广泛印行署名可查印刷史和Hancock文字资料；保存文字工具返回关于Dunlap broadside及署名的信息，画中接收者识别不是唯一信息来源。'),
 'q_026645': ('answer_only', 'Toraja舞蹈→传统屋顶装饰边色', 'red可能来自传统建筑记忆或常见装饰色，消融也命中但没有可靠目标屋顶trim的文字描述。不能把Tongkonan整体颜色资料当特定屋脊边线证据。'),
 'q_026750': ('answer_only', 'El Paso→山上灯星→独立灯泡或灯带', '二选一有较高猜中概率。GPT答individual bulbs，消融却错当neon tubing；未在现有轨迹取得灯星构造文本，视觉必要性未证伪。'),
 'q_027076': ('answer_only', 'Wang Ming→Comintern group photo邻座腿姿', 'crossed legs是常见坐姿，消融答对但未验证哪一位邻座及正确腿姿文字。需要正确人物/照片的证据，不能由二选式姿态命中证明可文本替代。'),
 'q_027110': ('question_shortcut', 'Copacabana唱片署名→原制作剧院与时长', '问题已经给June 1994/Prince of Wales Theatre，末问又要theatre，剧院部分直接泄露。演出时长是文本剧史而非图像属性；two years金标与其他轨迹15 months冲突需单独核对。'),
 'q_027556': ('text_evidence', 'Michigan球衣logo→Big Ten→冠军制度史', '消融文字给Bowl Coalition/Alliance/BCS序列。末问本身描述旧冠军产生制度，球衣conference logo并非回答这段通用制度史的必需线索。'),
 'q_027790': ('text_evidence', 'Livingstone Channel照片publisher→工艺名', '文本消融已识别Detroit Publishing Company；公司工艺名可用文本公司史核对。GPT答Phostint却因金标photostint被判错，美国国会图书馆书目使用Phostint，金标拼写疑似有误，不能简单判模型错。'),
 'q_027882': ('text_evidence', 'Edward VI画中压人书→Hebrew Bible版本', '完整轨迹仅一次文本搜索就返回Leningrad Codex最早完整Hebrew Bible。问题对宗教书的描述及末问极接近该常识，画中书不提供独有版本信息。'),
 'q_028071': ('text_evidence', 'Virgil Mosaic侧女神→Clio象征物', '消融turn11直接给trumpet与clepsydra；问题也预先提示timekeeping/proclamation，答案已受语义强约束，不必观察mosaic识別物件。'),
 'q_028154': ('answer_only', 'Benigni抱Berlinguer→鞋色', 'black是男性正式鞋极强先验，GPT一词命中，消融无判分。现有证据没有该照片鞋色的文字来源，属于易猜视觉题而非已证实文本题。'),
 'q_028175': ('answer_only', 'Vercingetorix画→Roman standard顶端动物', 'Roman aquila鹰符有强常识先验，但题目限定最高旗杆，需要目标画对应。现有无搜图答对不能区分图像记忆、常识或文字依据，不能称特定位置已验证。'),
 'q_028229': ('answer_only', 'Solovetsky防御圆塔→屋顶几何', '圆塔加屋顶常可猜conical；消融缺判分且无正确目标文字说明。支持低区分力疑虑，不足以证明该屋顶细节可文本取得。'),
 'q_028295': ('text_evidence', 'Lincoln→Fort Sumter战斗图击中结构', '消融turn6/18战役史正文明确shell strikes the flagpole，历史叙述直接提供画中对应事件，图片不是获取flagpole答案的唯一通道。'),
 'q_028499': ('text_evidence', 'Jahangir画中James I→岳父Frederick II', '消融turn9等人物生平明确dyslexia；James I和岳父关系可由文本查。GPT也只答Dyslexia却被judge以背景不足判错，标签对短答案不一致。'),
 'q_028584': ('answer_only', 'Merckx小时纪录自行车→车把带颜色', '白色车把带可能是该名车记忆，消融缺判分，未有描述tape color的文字证据。不能以GPT答案认定可以跳过目标自行车照片。'),
 'q_028652': ('answer_only', 'Bhore报告封面→标题分行数', 'read_url turn7明确只有标题文字，无法知道PDF封面分行；turn11也是一行文本。模型自行排成4行后答对，属于布局猜测或记忆，尚无排版文字证据。'),
 'q_028689': ('text_evidence', 'Soohorang→Bandabi胸前图案', '消融turn4/6官方发布文字解释Bandabi名字来自half-moon并写白色crescent胸章。吉祥物固定设计有文字描述，无需寻找unveiling图片。'),
 'q_028763': ('unverified_bypass', 'Wombling Free海报女演员→Tony获奖剧', '电影演员表和女演员生平可能代替海报署名位置，但保存文字轨迹未闭合正确名字及The History Boys。GPT无输入图答对仍只是一条经验结果，不能确认具体海报位置可文本解。'),
 'q_028877': ('quality_problem', 'Stratocaster→Tom Morello→Brad Wilk', '问题把drummer与guitarist合为同一限定，和标注人物关系不一致。应因关系混乱重写/过滤，不能把失败或语法错误当无需视觉的证明。'),
 'q_029008': ('text_evidence', 'Matthew Paris地图WALLIA→Wales词源', '问题已经写WALLIA，地域可按名称对应Wales；消融文字给Volcae→Walhaz词源，地图和画家识别对词源答案不具独有作用。'),
 'q_029050': ('text_evidence', 'Kendrick 2015封面→法官眼睛符号', '消融turn16百科图像替代文本写eyes crossed out，turn18/20正文再说明crosses over eyes。此处是清楚的视觉属性被网页文字描述，不只是模型猜X。'),
 'q_029169': ('unverified_bypass', 'Jan Dara海报节展桂冠→Tokyo Grand Prix年份', 'Jan Dara参展文字可能说明Tokyo，但片子可参加多个电影节；不能保证文字资料中任一节展就是该版海报桂冠。现有文本错到Seattle，原过滤结论应暂缓。'),
 'q_029200': ('unverified_bypass', 'Pressburg后会面画→Francis II/Napoleon亲属关系', '亲属关系和展览日期都可查文本，但需先定位具体画作、展出时点，不能因为father-in-law/son-in-law是著名关系就称相关图片必然冗余。现有证据以无图GPT答对为主。'),
 'q_029219': ('text_evidence', '1992丹麦守门员庆祝图→Peter Schmeichel derby纪录', '问题丹麦1992夺冠、goalkeeper、两家Manchester俱乐部已强约束Schmeichel；消融turn4直接读到他未输derby的记录，庆祝图无需提供独有信息。'),
 'q_029437': ('unverified_bypass', 'London Protocol地图红色飞地→East Prussia人口比例', 'geography或地图文字说明可能足以给East Prussia，但保存运行错到Saarland/不同territory百分比；未拿到20.36%正确文本表。可能绕过而非已验证绕过，应暂缓过滤。'),
 'q_029463': ('answer_only', 'Lincoln-Douglas书标题页→名字后标点', '书目文字经常规范化标点，与1860特定title page原版不等价；GPT答comma也可能按名字排版猜。无原版排印文本证据前不能称无需看标题页。'),
 'q_029673': ('text_evidence', 'Kabul C17照片→Soviet withdrawal→40th Army指挥官', '问题直接问40th Army退军负责人；消融turn8文本搜索给Boris Gromov，Kabul飞机图片和前置历史链与目标人物无独有关系。'),
 'q_030017': ('answer_only', 'Arlen Specter→John Roberts领带点纹', '此处答对的是RL5无搜图消融，GPT-5答Paisley、32B完整工具答diagonal stripes都错。RL5多轮查Neera Tanden而非John Roberts，turn44触发max_turns_finalization；turn45明确文字未给纹样但必须交最终答案，按most likely推断small repeating dots，随后判对。属于错误人物链上的强制收尾推断命中，不能作为文字替代证据。'),
 'q_030172': ('answer_only', 'YBC 7289→Yale collection→YBC 4644缺损角', '四角选择只有有限候选，GPT答upper-right，消融无判分。缺损部位是物件特定细节，尚无藏品文字说明或其他可靠绕过证据。'),
 'q_030200': ('text_evidence', 'Dr Strangelove核弹骑手→Slim Pickens艺名', '问题中riding nuclear bomb已明确影片标志场景；消融turn5生平给rodeo manager的slim pickins及假名起因，另搜电影帧不必要。'),
 'q_030295': ('unverified_bypass', '2024 USA领奖图最左球员→两度效力俱乐部', 'GPT无输入图答Boston Breakers，但位置身份没有实际文字证据。可能按球员履历组合反推，也可能猜中；不能直接把特定领奖位置视为已由文本识别。'),
 'q_030351': ('answer_only', 'ARA Santa Fe裂口→暴露内部颜色', 'yellow是特定损伤图的局部属性。GPT一词命中且消融缺判分，现有文本没有对应裂口颜色描述，仍可能需要目标图。'),
 'q_030354': ('answer_only', 'Hall of Mirrors动物→Élysée门饰侧部纹样', 'diamond lattice是典型铁艺/栏杆纹样，GPT命中与消融命中可能来自常见样式推断；未获取正确ornament指定两侧纹样的文字证据。'),
 'q_030420': ('text_evidence', 'Eton学者肖像大学字→Oxford增长原因', '消融turn11文字给Henry II的1167禁赴Paris。院长/学者大学履历也可文本查，末端是已被文字直接描述的教育史事实，肖像inscription不是唯一通道。'),
 'q_031201': ('answer_only', 'Henry II/Cunigunde合葬墓→脚边托盾动物', 'lion是欧洲墓像脚边动物的强先验，GPT仅一词命中，消融无判分。未见该合葬墓emperor feet shield局部的文字说明，不能确认视觉可绕过。'),
 'q_031502': ('text_evidence', 'Odysseus绑桅杆画→Sirens→词源', '题干直接描述Odysseus/siren标志情节，消融文本给seira/cord/binders词源；不必搜画确认winged creatures身份。'),
}

GROUP_NAMES = {
 'text_evidence': '已有文本替代证据', 'question_shortcut': '题干直接绕过/泄露',
 'input_answer': '给定图片已含答案', 'answer_only': '仅有无搜图命中，机制未证实',
 'unverified_bypass': '文字替代路径尚未证实', 'judge_uncertain': '正确标签存疑',
 'quality_problem': '质量问题，不能归因为无需视觉',
}
KEYWORDS = {
 'q_000015':['中海之星'], 'q_001393':['DXSA 7206'], 'q_011890':['seven years'],
 'q_012243':['Toonahowi'], 'q_013742':['crossed out the pre-printed'],
 'q_020564':['Drummond'], 'q_022987':['condoms','contraceptives'],
 'q_023335':['Invitation to the Dance'], 'q_023371':['Stamfordham'],
 'q_024662':['First Day in Hell'], 'q_025108':['first triplicate'],
 'q_025769':['devotion to parents'], 'q_026458':['Hancock','Dunlap'],
 'q_027556':['Bowl Coalition'], 'q_027882':['Leningrad'],
 'q_028071':['Clepsydra'], 'q_028499':['dyslexia'], 'q_028689':['crescent'],
 'q_029008':['Volcae'], 'q_029050':['eyes crossed out','crosses placed'],
 'q_029219':['unbeaten record'], 'q_029673':['Gromov'],
 'q_030200':['doubtful rodeo manager'], 'q_030420':['1167'],
 'q_031502':['seira'], 'q_028295':['shell strikes the flagpole'],
}


def read_rows(p):
 return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def without_query(output):
 raw = re.sub(r'</?tool_response>', '', str(output)).strip()
 try:
  obj = json.loads(raw)
 except ValueError:
  return raw
 def clean(v):
  if isinstance(v, dict): return {k:clean(x) for k,x in v.items() if k not in ('query','goal')}
  if isinstance(v, list): return [clean(x) for x in v]
  return v
 return json.dumps(clean(obj), ensure_ascii=False)


def main():
 previous = [r for r in read_rows(ROOT/'review.jsonl') if r['category']=='image-interleave' and r['decision']=='filter']
 evidence = {r['case_id']:r for r in read_rows(ROOT/'evidence.jsonl')}
 assert len(previous)==81 and {r['case_id'] for r in previous}==set(NOTES)
 out = []
 for r in previous:
  cid=r['case_id']; group, bridge, note=NOTES[cid]
  anchors=[]
  for source in ('ablation_calls','baseline_calls'):
   for c in evidence[cid][source]:
    if c['name'] not in ('t2t_search','read_url'): continue
    text=without_query(c['output'])
    for word in KEYWORDS.get(cid,[]):
     pos=text.casefold().find(word.casefold())
     if pos>=0:
      anchors.append({'trace':source,'file':r['sources']['ablation' if source=='ablation_calls' else 'baseline'],
                      'turn':c['turn'],'tool':c['name'],'excerpt':text[max(0,pos-130):pos+len(word)+240]})
      break
  if cid=='q_027790':
   anchors.append({'source_url':'https://www.loc.gov/pictures/collection/det/bibliography.html',
                   'evidence':'Library of Congress bibliography lists Detroit Publishing Company. Little "Phostint" Journeys. Detroit, 1912-.',
                   'scope':'Verifies spelling Phostint in the publisher\'s titles; not an independent verification of the question\'s 1907 date.'})
  if group=='text_evidence': assert anchors,cid
  recommendation = ('支持过滤（主动搜图不必要）' if group in ('text_evidence','question_shortcut','input_answer') else
                    '支持质量过滤/改写，视觉必要性另论' if group=='quality_problem' else
                    '暂缓确定性过滤，复核判分/金标' if group=='judge_uncertain' else
                    '可按无搜图规则排除，但不足以认定视觉不必要' if group=='answer_only' else
                    '暂缓过滤，文字绕过未证实')
  item={k:r[k] for k in ('case_id','question','reference_answer','input_image_count','gpt5_acc','gpt5_answer','ablation_acc','ablation_answer','reason_code','sources')}
  item.update(evidence_group=group, evidence_group_name=GROUP_NAMES[group],image_bridge=bridge,
              detailed_recheck=note,recommendation=recommendation,text_evidence_anchors=anchors[:6])
  out.append(item)
 OUT.mkdir(parents=True,exist_ok=True)
 (OUT/'cases.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in out))
 fields=list(out[0])
 with (OUT/'cases.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in out:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
 counts=Counter(r['evidence_group'] for r in out)
 stats={'total_previously_filtered':81,'actual_input_image_counts':dict(Counter(r['input_image_count'] for r in out)),
        'evidence_groups':dict(counts),'new_inference_calls':0,
        'input_images_actually_viewed':['q_004785','q_023533'],
        'old_filter_manifest_changed':False,
        'interpretation':'No-tools can see the initial image. Success without t2i is not evidence of no visual dependence. Text evidence is from saved tool outputs, with source links; not independent ground-truth adjudication.'}
 stats['recommended_under_original_screening_rule'] = {
  'continue_filter': sum(r['evidence_group'] not in ('unverified_bypass','judge_uncertain') for r in out),
  'defer_and_recheck': sum(r['evidence_group'] in ('unverified_bypass','judge_uncertain') for r in out),
  'note':'31 answer_only cases qualify under the original empirical rule but lack a demonstrated textual replacement; this is not a causal necessity verdict.'}
 (OUT/'summary.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n')
 selections = {
  'supported_t2i_filter_ids': {'text_evidence','question_shortcut','input_answer'},
  'quality_filter_ids': {'quality_problem'},
  'empirical_rule_only_ids': {'answer_only'},
  'defer_filter_ids': {'unverified_bypass','judge_uncertain'},
 }
 for name,groups in selections.items():
  (OUT/(name+'.txt')).write_text('\n'.join(r['case_id'] for r in out if r['evidence_group'] in groups)+'\n')
 lines=['# image-interleave过滤样本逐条复查','',
        '81条过滤样本全部复查。65条有输入图，16条无输入图。不能把no-tools解释为text-only。',
        '本轮使用保存的问题、标注路径、无工具回答、文字消融工具返回和完整轨迹；另查看2张原始输入图片，并核对1项外部书目。未新增模型推理。','',
        '|证据组|数量|','|---|---:|']
 lines += [f'|{GROUP_NAMES[k]}|{v}|' for k,v in counts.items()]
 lines += ['', '有文本工具返回的答案/属性，比仅有正确标签更有说服力；仍需注意原始网页提取和judge可能出错。',
           '题干可直接查末端问题，并不需要中间图片；这种结构性绕过不等同于证明所有相关图片无用。',
           'answer_only组的过滤只符合用户给定的“no-tools/禁用搜图答对即排除”经验规则，不能写成已证实不需要视觉。',
           'unverified_bypass应暂缓确定过滤。judge_uncertain应先复核物件类别/答案完整性。quality_problem按定位/关系质量过滤，而非视觉冗余。',
           '按原经验筛选规则，建议66条继续过滤、15条暂缓；其中31条继续过滤仅依据无搜图命中记录，不能称已证实文本可替代视觉。',
           '原81条过滤清单未自动改写；本目录给出复查结果和修正建议。','',
           '外部拼写核对：美国国会图书馆书目记载出版物Little "Phostint" Journeys。该资料支持Phostint拼写，并不验证1907日期。',
           '[Library of Congress bibliography](https://www.loc.gov/pictures/collection/det/bibliography.html)','']
 for r in out:
  lines += [f"## {r['case_id']} — {r['evidence_group_name']}", '',
            f"图片桥接：{r['image_bridge']}", '',f"问题：{r['question']}", '',
            f"金标：{r['reference_answer']}", '',f"复查：{r['detailed_recheck']}", '',
            f"建议：{r['recommendation']}", '']
 (OUT/'report.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(stats,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
