#import <UIKit/UIKit.h>
#import <PhotosUI/PhotosUI.h>
#include <stdatomic.h>
#import "KronosThemeAndModels.h"
#import "KronosWorkspaceAndVoice.h"

typedef struct {
    __unsafe_unretained NSMutableString *buffer;
    __unsafe_unretained void (^onFlush)(NSString *snapshot);
    atomic_bool *cancelFlag;
    double lastFlushMs;
} KronosStreamCtx;

static bool kronos_chat_stream_cb(const char *utf8_ptr, size_t utf8_len, void *user_data) {
    if (utf8_ptr == NULL || user_data == NULL) return false;
    KronosStreamCtx *ctx = (KronosStreamCtx *)user_data;
    if (ctx->cancelFlag != NULL && atomic_load(ctx->cancelFlag)) return false;
    NSString *piece = [[NSString alloc] initWithBytes:utf8_ptr length:utf8_len encoding:NSUTF8StringEncoding];
    if (piece != nil) [ctx->buffer appendString:piece];
    double now = kronos_monotonic_ms();
    if (now - ctx->lastFlushMs >= 32.0) {
        ctx->lastFlushMs = now;
        if (ctx->onFlush != nil) ctx->onFlush([ctx->buffer copy]);
    }
    return true;
}

static inline NSString *kronosCleanDraftOutput(NSString *raw) {
    if (raw.length == 0) return @"";
    NSString *dblStar = [NSString stringWithFormat:@"%c%c", 42, 42];
    NSString *excl = [NSString stringWithFormat:@"%c", 33];
    NSString *s = [raw stringByReplacingOccurrencesOfString:dblStar withString:@""];
    s = [s stringByReplacingOccurrencesOfString:excl withString:@"."];
    s = [s stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
    NSArray<NSString *> *prefixes = @[@"Message:", @"Draft:", @"WhatsApp message:", @"LinkedIn post:", @"Post:"];
    for (NSString *p in prefixes) {
        if ([[s lowercaseString] hasPrefix:[p lowercaseString]]) {
            s = [[s substringFromIndex:p.length] stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
        }
    }
    if (s.length >= 2 && [s hasPrefix:@"\""] && [s hasSuffix:@"\""]) {
        s = [[s substringWithRange:NSMakeRange(1, s.length - 2)] stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
    }
    return s;
}

@interface KronosChatViewController : UIViewController <
    UITextViewDelegate,
    UITableViewDataSource,
    UITableViewDelegate,
    UITextFieldDelegate,
    PHPickerViewControllerDelegate,
    UIDocumentPickerDelegate,
    UIImagePickerControllerDelegate,
    UINavigationControllerDelegate,
    KronosPasteAttachmentDelegate,
    KronosVoiceControllerDelegate,
    KronosSettingsViewControllerDelegate
>
@property (nonatomic, strong) UIButton *sidebarButton;
@property (nonatomic, strong) UILabel *headerTitleLabel;
@property (nonatomic, strong) UILabel *headerSubtitleLabel;
@property (nonatomic, strong) UIButton *ttsToggleButton;
@property (nonatomic, strong) UIButton *settingsButton;
@property (nonatomic, strong) UIButton *plusNewChatButton;

@property (nonatomic, strong) UIScrollView *scrollView;
@property (nonatomic, strong) UIStackView *messagesStack;
@property (nonatomic, strong) UIView *welcomeView;

@property (nonatomic, strong) UIView *composerContainer;
@property (nonatomic, strong) UIScrollView *attachmentsStrip;
@property (nonatomic, strong) UIStackView *attachmentsStack;
@property (nonatomic, strong) NSLayoutConstraint *attachmentsHeightConstraint;
@property (nonatomic, strong) UIButton *attachButton;
@property (nonatomic, strong) KronosInputTextView *inputTextView;
@property (nonatomic, strong) UILabel *placeholderLabel;
@property (nonatomic, strong) NSLayoutConstraint *inputHeightConstraint;
@property (nonatomic, strong) UIButton *micButton;
@property (nonatomic, strong) UIButton *sendButton;

@property (nonatomic, strong) UIView *sidebarBackdrop;
@property (nonatomic, strong) UIView *sidebarPanel;
@property (nonatomic, strong) NSLayoutConstraint *sidebarLeadingConstraint;
@property (nonatomic, strong) UITextField *sidebarSearchField;
@property (nonatomic, strong) UILabel *workspaceCountBadge;
@property (nonatomic, strong) UITableView *conversationsTable;
@property (nonatomic, strong) UILabel *sidebarSettingsSubtitle;
@property (nonatomic, assign) BOOL sidebarOpen;

@property (nonatomic, strong) NSMutableArray<NSMutableDictionary *> *conversations;
@property (nonatomic, strong) NSMutableArray<NSMutableDictionary *> *filteredConversations;
@property (nonatomic, strong) NSMutableDictionary *activeConversation;
@property (nonatomic, strong) NSMutableArray<KronosAttachmentItem *> *pendingAttachments;

@property (nonatomic, copy) NSString *pendingWhatsAppRecipient;
@property (nonatomic, assign) BOOL pendingWhatsAppAwaitingRecipient;
@property (nonatomic, strong) NSURL *lastWhatsAppDraftURL;
@property (nonatomic, copy) NSString *lastLinkedInDraftText;
@property (nonatomic, strong) UIImage *lastLinkedInDraftPhoto;

@property (nonatomic, strong) KronosVoiceController *voiceController;
@property (nonatomic, strong) dispatch_queue_t engineQueue;
@property (nonatomic, assign) KronosMobileSession *session;
@property (nonatomic, copy) NSString *kvSessionConversationId;
@property (nonatomic, assign) NSUInteger kvWindowBaseTurnIndex;
@property (nonatomic, assign) BOOL modelReady;
@property (nonatomic, assign) BOOL isGenerating;
@property (nonatomic, assign) BOOL lastTurnWasSpoken;
@end

@implementation KronosChatViewController {
    atomic_bool _cancelFlag;
}

static const CGFloat kSidebarWidth = 296.0;

- (void)dealloc {
    KronosMobileSession *s = _session;
    _session = NULL;
    if (s != NULL) kronos_session_destroy(s);
}

- (void)viewDidLoad {
    [super viewDidLoad];
    atomic_init(&_cancelFlag, false);
    self.view.backgroundColor = kronosBgColor();
    self.engineQueue = dispatch_queue_create("com.ipmscoutek.kronos.engine", DISPATCH_QUEUE_SERIAL);
    self.pendingAttachments = [NSMutableArray array];

    self.voiceController = [[KronosVoiceController alloc] init];
    self.voiceController.delegate = self;

    self.conversations = [KronosConversationStore loadConversations];
    kronosBackfillLearnedMemoryFromConversations(self.conversations);
    kronosSyncContinualMemoryAndDistillVaultWithMac();
    if (self.conversations.count == 0) {
        NSMutableDictionary *blank = [KronosConversationStore createBlankConversation];
        [self.conversations addObject:blank];
    }
    self.activeConversation = self.conversations.firstObject;
    self.filteredConversations = [self.conversations mutableCopy];

    [self buildTopNavigationBar];
    [self buildBottomComposer];
    [self buildTranscriptArea];
    [self buildSlideOverSidebar];
    [self setupEdgeSwipeGestures];

    [self syncTtsButtonAppearance];
    [self refreshSidebarSettingsFooter];
    [self renderActiveConversationTranscript];
    [self refreshWorkspaceFileCount];
    [self loadLocalEngineInBackground];

    if ([KronosMobileSettings sharedSettings].developDeviceContext) {
        __weak typeof(self) weakSelf = self;
        [[KronosIosDeviceContextManager sharedManager] refreshDeviceContextInteractive:NO completion:^{
            [weakSelf refreshSidebarSettingsFooter];
        }];
    }
}

#pragma mark - Layout Construction

- (void)buildTopNavigationBar {
    UIView *bar = [[UIView alloc] init];
    bar.translatesAutoresizingMaskIntoConstraints = NO;
    bar.backgroundColor = kronosBgColor();
    bar.tag = 9001;
    [self.view addSubview:bar];

    UIView *hairline = [[UIView alloc] init];
    hairline.translatesAutoresizingMaskIntoConstraints = NO;
    hairline.backgroundColor = kronosLineColor();
    [bar addSubview:hairline];

    self.sidebarButton = [UIButton buttonWithType:UIButtonTypeSystem];
    self.sidebarButton.translatesAutoresizingMaskIntoConstraints = NO;
    UIImageSymbolConfiguration *symCfg = [UIImageSymbolConfiguration configurationWithPointSize:17.0 weight:UIImageSymbolWeightMedium];
    [self.sidebarButton setImage:[UIImage systemImageNamed:@"sidebar.left" withConfiguration:symCfg] forState:UIControlStateNormal];
    self.sidebarButton.tintColor = kronosTextColor();
    self.sidebarButton.accessibilityLabel = @"Toggle conversations sidebar";
    [self.sidebarButton addTarget:self action:@selector(toggleSidebarTapped) forControlEvents:UIControlEventTouchUpInside];
    [bar addSubview:self.sidebarButton];

    UIStackView *centerStack = [[UIStackView alloc] init];
    centerStack.translatesAutoresizingMaskIntoConstraints = NO;
    centerStack.axis = UILayoutConstraintAxisVertical;
    centerStack.alignment = UIStackViewAlignmentCenter;
    centerStack.spacing = 1.0;
    [bar addSubview:centerStack];

    self.headerTitleLabel = [[UILabel alloc] init];
    self.headerTitleLabel.text = @"Kronos";
    self.headerTitleLabel.textColor = kronosTextColor();
    self.headerTitleLabel.font = [UIFont systemFontOfSize:16.5 weight:UIFontWeightSemibold];
    self.headerTitleLabel.numberOfLines = 1;
    self.headerTitleLabel.lineBreakMode = NSLineBreakByTruncatingTail;
    [self.headerTitleLabel setContentCompressionResistancePriority:UILayoutPriorityDefaultLow forAxis:UILayoutConstraintAxisHorizontal];
    [centerStack addArrangedSubview:self.headerTitleLabel];

    self.headerSubtitleLabel = [[UILabel alloc] init];
    self.headerSubtitleLabel.text = @"Starting up...";
    self.headerSubtitleLabel.textColor = kronosMutedColor();
    self.headerSubtitleLabel.font = [UIFont systemFontOfSize:11.0 weight:UIFontWeightRegular];
    self.headerSubtitleLabel.numberOfLines = 1;
    self.headerSubtitleLabel.lineBreakMode = NSLineBreakByTruncatingTail;
    [self.headerSubtitleLabel setContentCompressionResistancePriority:UILayoutPriorityDefaultLow forAxis:UILayoutConstraintAxisHorizontal];
    [centerStack addArrangedSubview:self.headerSubtitleLabel];

    self.ttsToggleButton = [UIButton buttonWithType:UIButtonTypeSystem];
    self.ttsToggleButton.translatesAutoresizingMaskIntoConstraints = NO;
    UIImageSymbolConfiguration *spkCfg = [UIImageSymbolConfiguration configurationWithPointSize:15.0 weight:UIImageSymbolWeightMedium];
    [self.ttsToggleButton setImage:[UIImage systemImageNamed:@"speaker.slash" withConfiguration:spkCfg] forState:UIControlStateNormal];
    self.ttsToggleButton.tintColor = kronosMutedColor();
    self.ttsToggleButton.accessibilityLabel = @"Toggle spoken voice replies";
    [self.ttsToggleButton addTarget:self action:@selector(ttsToggleTapped) forControlEvents:UIControlEventTouchUpInside];
    [bar addSubview:self.ttsToggleButton];

    self.settingsButton = [UIButton buttonWithType:UIButtonTypeSystem];
    self.settingsButton.translatesAutoresizingMaskIntoConstraints = NO;
    UIImageSymbolConfiguration *gearCfg = [UIImageSymbolConfiguration configurationWithPointSize:15.5 weight:UIImageSymbolWeightMedium];
    [self.settingsButton setImage:[UIImage systemImageNamed:@"gearshape" withConfiguration:gearCfg] forState:UIControlStateNormal];
    self.settingsButton.tintColor = kronosMutedColor();
    self.settingsButton.accessibilityLabel = @"Settings and Device Context";
    [self.settingsButton addTarget:self action:@selector(openSettingsTapped) forControlEvents:UIControlEventTouchUpInside];
    [bar addSubview:self.settingsButton];

    self.plusNewChatButton = [UIButton buttonWithType:UIButtonTypeSystem];
    self.plusNewChatButton.translatesAutoresizingMaskIntoConstraints = NO;
    UIImageSymbolConfiguration *plusCfg = [UIImageSymbolConfiguration configurationWithPointSize:15.5 weight:UIImageSymbolWeightSemibold];
    [self.plusNewChatButton setImage:[UIImage systemImageNamed:@"plus" withConfiguration:plusCfg] forState:UIControlStateNormal];
    self.plusNewChatButton.tintColor = kronosTextColor();
    self.plusNewChatButton.backgroundColor = kronosSubtleColor();
    self.plusNewChatButton.layer.cornerRadius = 17.0;
    self.plusNewChatButton.accessibilityLabel = @"New conversation";
    [self.plusNewChatButton addTarget:self action:@selector(newConversationTapped) forControlEvents:UIControlEventTouchUpInside];
    [bar addSubview:self.plusNewChatButton];

    UILayoutGuide *safe = self.view.safeAreaLayoutGuide;
    NSLayoutConstraint *centerX = [centerStack.centerXAnchor constraintEqualToAnchor:bar.centerXAnchor];
    centerX.priority = UILayoutPriorityDefaultHigh;

    [NSLayoutConstraint activateConstraints:@[
        [bar.topAnchor constraintEqualToAnchor:self.view.topAnchor],
        [bar.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor],
        [bar.trailingAnchor constraintEqualToAnchor:self.view.trailingAnchor],

        [self.sidebarButton.topAnchor constraintEqualToAnchor:safe.topAnchor constant:4.0],
        [self.sidebarButton.leadingAnchor constraintEqualToAnchor:safe.leadingAnchor constant:8.0],
        [self.sidebarButton.widthAnchor constraintEqualToConstant:40.0],
        [self.sidebarButton.heightAnchor constraintEqualToConstant:40.0],
        [self.sidebarButton.bottomAnchor constraintEqualToAnchor:bar.bottomAnchor constant:-6.0],

        centerX,
        [centerStack.centerYAnchor constraintEqualToAnchor:self.sidebarButton.centerYAnchor],
        [centerStack.leadingAnchor constraintGreaterThanOrEqualToAnchor:self.sidebarButton.trailingAnchor constant:6.0],
        [centerStack.trailingAnchor constraintLessThanOrEqualToAnchor:self.ttsToggleButton.leadingAnchor constant:-6.0],

        [self.plusNewChatButton.centerYAnchor constraintEqualToAnchor:self.sidebarButton.centerYAnchor],
        [self.plusNewChatButton.trailingAnchor constraintEqualToAnchor:safe.trailingAnchor constant:-10.0],
        [self.plusNewChatButton.widthAnchor constraintEqualToConstant:34.0],
        [self.plusNewChatButton.heightAnchor constraintEqualToConstant:34.0],

        [self.settingsButton.centerYAnchor constraintEqualToAnchor:self.sidebarButton.centerYAnchor],
        [self.settingsButton.trailingAnchor constraintEqualToAnchor:self.plusNewChatButton.leadingAnchor constant:-5.0],
        [self.settingsButton.widthAnchor constraintEqualToConstant:34.0],
        [self.settingsButton.heightAnchor constraintEqualToConstant:34.0],

        [self.ttsToggleButton.centerYAnchor constraintEqualToAnchor:self.sidebarButton.centerYAnchor],
        [self.ttsToggleButton.trailingAnchor constraintEqualToAnchor:self.settingsButton.leadingAnchor constant:-2.0],
        [self.ttsToggleButton.widthAnchor constraintEqualToConstant:34.0],
        [self.ttsToggleButton.heightAnchor constraintEqualToConstant:34.0],

        [hairline.leadingAnchor constraintEqualToAnchor:bar.leadingAnchor],
        [hairline.trailingAnchor constraintEqualToAnchor:bar.trailingAnchor],
        [hairline.bottomAnchor constraintEqualToAnchor:bar.bottomAnchor],
        [hairline.heightAnchor constraintEqualToConstant:0.5],
    ]];
}

- (void)buildBottomComposer {
    self.composerContainer = [[UIView alloc] init];
    self.composerContainer.translatesAutoresizingMaskIntoConstraints = NO;
    self.composerContainer.backgroundColor = kronosBgColor();
    [self.view addSubview:self.composerContainer];

    self.attachmentsStrip = [[UIScrollView alloc] init];
    self.attachmentsStrip.translatesAutoresizingMaskIntoConstraints = NO;
    self.attachmentsStrip.showsHorizontalScrollIndicator = NO;
    self.attachmentsStrip.hidden = YES;
    [self.composerContainer addSubview:self.attachmentsStrip];

    self.attachmentsStack = [[UIStackView alloc] init];
    self.attachmentsStack.translatesAutoresizingMaskIntoConstraints = NO;
    self.attachmentsStack.axis = UILayoutConstraintAxisHorizontal;
    self.attachmentsStack.spacing = 8.0;
    [self.attachmentsStrip addSubview:self.attachmentsStack];

    UIView *capsule = [[UIView alloc] init];
    capsule.translatesAutoresizingMaskIntoConstraints = NO;
    capsule.backgroundColor = kronosPanelColor();
    capsule.layer.cornerRadius = 23.0;
    capsule.layer.borderWidth = 1.0;
    capsule.layer.borderColor = kronosLineColor().CGColor;
    [self.composerContainer addSubview:capsule];

    UIImageSymbolConfiguration *iconCfg = [UIImageSymbolConfiguration configurationWithPointSize:18.0 weight:UIImageSymbolWeightMedium];

    self.attachButton = [UIButton buttonWithType:UIButtonTypeSystem];
    self.attachButton.translatesAutoresizingMaskIntoConstraints = NO;
    [self.attachButton setImage:[UIImage systemImageNamed:@"plus" withConfiguration:iconCfg] forState:UIControlStateNormal];
    self.attachButton.tintColor = kronosMutedColor();
    self.attachButton.backgroundColor = kronosSubtleColor();
    self.attachButton.layer.cornerRadius = 17.0;
    self.attachButton.accessibilityLabel = @"Attach photo, file, or clipboard";
    [self.attachButton addTarget:self action:@selector(attachTapped) forControlEvents:UIControlEventTouchUpInside];
    [capsule addSubview:self.attachButton];

    self.inputTextView = [[KronosInputTextView alloc] init];
    self.inputTextView.translatesAutoresizingMaskIntoConstraints = NO;
    self.inputTextView.backgroundColor = [UIColor clearColor];
    self.inputTextView.textColor = kronosTextColor();
    self.inputTextView.tintColor = kronosAccentColor();
    self.inputTextView.font = [UIFont systemFontOfSize:16.0 weight:UIFontWeightRegular];
    self.inputTextView.scrollEnabled = NO;
    self.inputTextView.textContainerInset = UIEdgeInsetsMake(9, 4, 9, 4);
    self.inputTextView.delegate = self;
    self.inputTextView.attachmentPasteDelegate = self;
    [capsule addSubview:self.inputTextView];

    self.placeholderLabel = [[UILabel alloc] init];
    self.placeholderLabel.translatesAutoresizingMaskIntoConstraints = NO;
    self.placeholderLabel.text = @"Ask Kronos...";
    self.placeholderLabel.textColor = kronosMutedColor();
    self.placeholderLabel.font = [UIFont systemFontOfSize:16.0 weight:UIFontWeightRegular];
    [capsule addSubview:self.placeholderLabel];

    self.micButton = [UIButton buttonWithType:UIButtonTypeSystem];
    self.micButton.translatesAutoresizingMaskIntoConstraints = NO;
    [self.micButton setImage:[UIImage systemImageNamed:@"mic.fill" withConfiguration:iconCfg] forState:UIControlStateNormal];
    self.micButton.tintColor = kronosMutedColor();
    self.micButton.layer.cornerRadius = 17.0;
    self.micButton.accessibilityLabel = @"Speak to Kronos";
    [self.micButton addTarget:self action:@selector(micTapped) forControlEvents:UIControlEventTouchUpInside];
    [capsule addSubview:self.micButton];

    UIImageSymbolConfiguration *sendCfg = [UIImageSymbolConfiguration configurationWithPointSize:28.0 weight:UIImageSymbolWeightSemibold];
    self.sendButton = [UIButton buttonWithType:UIButtonTypeSystem];
    self.sendButton.translatesAutoresizingMaskIntoConstraints = NO;
    [self.sendButton setImage:[UIImage systemImageNamed:@"arrow.up.circle.fill" withConfiguration:sendCfg] forState:UIControlStateNormal];
    self.sendButton.tintColor = kronosAccentColor();
    self.sendButton.enabled = NO;
    self.sendButton.accessibilityLabel = @"Send message";
    [self.sendButton addTarget:self action:@selector(sendOrStopTapped) forControlEvents:UIControlEventTouchUpInside];
    [capsule addSubview:self.sendButton];

    self.attachmentsHeightConstraint = [self.attachmentsStrip.heightAnchor constraintEqualToConstant:0.0];
    self.inputHeightConstraint = [self.inputTextView.heightAnchor constraintEqualToConstant:38.0];
    UILayoutGuide *safe = self.view.safeAreaLayoutGuide;
    UILayoutGuide *kb = self.view.keyboardLayoutGuide;

    [NSLayoutConstraint activateConstraints:@[
        [self.composerContainer.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor],
        [self.composerContainer.trailingAnchor constraintEqualToAnchor:self.view.trailingAnchor],
        [self.composerContainer.bottomAnchor constraintEqualToAnchor:self.view.bottomAnchor],

        [self.attachmentsStrip.topAnchor constraintEqualToAnchor:self.composerContainer.topAnchor constant:6.0],
        [self.attachmentsStrip.leadingAnchor constraintEqualToAnchor:safe.leadingAnchor constant:12.0],
        [self.attachmentsStrip.trailingAnchor constraintEqualToAnchor:safe.trailingAnchor constant:-12.0],
        self.attachmentsHeightConstraint,

        [self.attachmentsStack.topAnchor constraintEqualToAnchor:self.attachmentsStrip.contentLayoutGuide.topAnchor],
        [self.attachmentsStack.bottomAnchor constraintEqualToAnchor:self.attachmentsStrip.contentLayoutGuide.bottomAnchor],
        [self.attachmentsStack.leadingAnchor constraintEqualToAnchor:self.attachmentsStrip.contentLayoutGuide.leadingAnchor],
        [self.attachmentsStack.trailingAnchor constraintEqualToAnchor:self.attachmentsStrip.contentLayoutGuide.trailingAnchor],
        [self.attachmentsStack.heightAnchor constraintEqualToAnchor:self.attachmentsStrip.frameLayoutGuide.heightAnchor],

        [capsule.topAnchor constraintEqualToAnchor:self.attachmentsStrip.bottomAnchor constant:6.0],
        [capsule.leadingAnchor constraintEqualToAnchor:safe.leadingAnchor constant:12.0],
        [capsule.trailingAnchor constraintEqualToAnchor:safe.trailingAnchor constant:-12.0],
        [capsule.bottomAnchor constraintEqualToAnchor:kb.topAnchor constant:-8.0],

        [self.attachButton.leadingAnchor constraintEqualToAnchor:capsule.leadingAnchor constant:6.0],
        [self.attachButton.bottomAnchor constraintEqualToAnchor:capsule.bottomAnchor constant:-6.0],
        [self.attachButton.widthAnchor constraintEqualToConstant:34.0],
        [self.attachButton.heightAnchor constraintEqualToConstant:34.0],

        [self.inputTextView.topAnchor constraintEqualToAnchor:capsule.topAnchor constant:4.0],
        [self.inputTextView.bottomAnchor constraintEqualToAnchor:capsule.bottomAnchor constant:-4.0],
        [self.inputTextView.leadingAnchor constraintEqualToAnchor:self.attachButton.trailingAnchor constant:6.0],
        [self.inputTextView.trailingAnchor constraintEqualToAnchor:self.micButton.leadingAnchor constant:-4.0],
        self.inputHeightConstraint,

        [self.placeholderLabel.leadingAnchor constraintEqualToAnchor:self.inputTextView.leadingAnchor constant:9.0],
        [self.placeholderLabel.centerYAnchor constraintEqualToAnchor:self.inputTextView.centerYAnchor],

        [self.micButton.trailingAnchor constraintEqualToAnchor:self.sendButton.leadingAnchor constant:-4.0],
        [self.micButton.bottomAnchor constraintEqualToAnchor:capsule.bottomAnchor constant:-6.0],
        [self.micButton.widthAnchor constraintEqualToConstant:34.0],
        [self.micButton.heightAnchor constraintEqualToConstant:34.0],

        [self.sendButton.trailingAnchor constraintEqualToAnchor:capsule.trailingAnchor constant:-5.0],
        [self.sendButton.bottomAnchor constraintEqualToAnchor:capsule.bottomAnchor constant:-5.0],
        [self.sendButton.widthAnchor constraintEqualToConstant:36.0],
        [self.sendButton.heightAnchor constraintEqualToConstant:36.0],
    ]];
}

- (void)buildTranscriptArea {
    UIView *bar = [self.view viewWithTag:9001];
    self.scrollView = [[UIScrollView alloc] init];
    self.scrollView.translatesAutoresizingMaskIntoConstraints = NO;
    self.scrollView.keyboardDismissMode = UIScrollViewKeyboardDismissModeInteractive;
    self.scrollView.alwaysBounceVertical = YES;
    [self.view addSubview:self.scrollView];

    UITapGestureRecognizer *tap = [[UITapGestureRecognizer alloc] initWithTarget:self action:@selector(dismissKeyboard)];
    tap.cancelsTouchesInView = NO;
    [self.scrollView addGestureRecognizer:tap];

    self.messagesStack = [[UIStackView alloc] init];
    self.messagesStack.translatesAutoresizingMaskIntoConstraints = NO;
    self.messagesStack.axis = UILayoutConstraintAxisVertical;
    self.messagesStack.spacing = 16.0;
    [self.scrollView addSubview:self.messagesStack];

    UILayoutGuide *cg = self.scrollView.contentLayoutGuide;
    UILayoutGuide *fg = self.scrollView.frameLayoutGuide;
    [NSLayoutConstraint activateConstraints:@[
        [self.scrollView.topAnchor constraintEqualToAnchor:bar.bottomAnchor],
        [self.scrollView.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor],
        [self.scrollView.trailingAnchor constraintEqualToAnchor:self.view.trailingAnchor],
        [self.scrollView.bottomAnchor constraintEqualToAnchor:self.composerContainer.topAnchor],

        [self.messagesStack.topAnchor constraintEqualToAnchor:cg.topAnchor constant:16.0],
        [self.messagesStack.leadingAnchor constraintEqualToAnchor:cg.leadingAnchor constant:16.0],
        [self.messagesStack.trailingAnchor constraintEqualToAnchor:cg.trailingAnchor constant:-16.0],
        [self.messagesStack.bottomAnchor constraintEqualToAnchor:cg.bottomAnchor constant:-20.0],
        [self.messagesStack.widthAnchor constraintEqualToAnchor:fg.widthAnchor constant:-32.0],
    ]];
}

- (void)buildSlideOverSidebar {
    self.sidebarBackdrop = [[UIView alloc] init];
    self.sidebarBackdrop.translatesAutoresizingMaskIntoConstraints = NO;
    self.sidebarBackdrop.backgroundColor = [UIColor colorWithWhite:0.0 alpha:0.48];
    self.sidebarBackdrop.alpha = 0.0;
    self.sidebarBackdrop.hidden = YES;
    UITapGestureRecognizer *closeTap = [[UITapGestureRecognizer alloc] initWithTarget:self action:@selector(closeSidebarAnimated)];
    [self.sidebarBackdrop addGestureRecognizer:closeTap];
    [self.view addSubview:self.sidebarBackdrop];

    self.sidebarPanel = [[UIView alloc] init];
    self.sidebarPanel.translatesAutoresizingMaskIntoConstraints = NO;
    self.sidebarPanel.backgroundColor = kronosSidebarBg();
    [self.view addSubview:self.sidebarPanel];

    UISwipeGestureRecognizer *swipeClose = [[UISwipeGestureRecognizer alloc] initWithTarget:self action:@selector(closeSidebarAnimated)];
    swipeClose.direction = UISwipeGestureRecognizerDirectionLeft;
    [self.sidebarPanel addGestureRecognizer:swipeClose];

    UILabel *brandLbl = [[UILabel alloc] init];
    brandLbl.translatesAutoresizingMaskIntoConstraints = NO;
    brandLbl.text = @"Kronos";
    brandLbl.textColor = kronosSidebarText();
    brandLbl.font = [UIFont systemFontOfSize:19.0 weight:UIFontWeightBold];
    [self.sidebarPanel addSubview:brandLbl];

    UIButton *closeBtn = [UIButton buttonWithType:UIButtonTypeSystem];
    closeBtn.translatesAutoresizingMaskIntoConstraints = NO;
    UIImageSymbolConfiguration *cfg = [UIImageSymbolConfiguration configurationWithPointSize:16.5 weight:UIImageSymbolWeightMedium];
    [closeBtn setImage:[UIImage systemImageNamed:@"sidebar.left" withConfiguration:cfg] forState:UIControlStateNormal];
    closeBtn.tintColor = kronosSidebarMuted();
    closeBtn.accessibilityLabel = @"Close sidebar";
    [closeBtn addTarget:self action:@selector(closeSidebarAnimated) forControlEvents:UIControlEventTouchUpInside];
    [self.sidebarPanel addSubview:closeBtn];

    UIButton *newConvBtn = [UIButton buttonWithType:UIButtonTypeSystem];
    newConvBtn.translatesAutoresizingMaskIntoConstraints = NO;
    [newConvBtn setTitle:@"  New conversation" forState:UIControlStateNormal];
    [newConvBtn setImage:[UIImage systemImageNamed:@"plus" withConfiguration:cfg] forState:UIControlStateNormal];
    newConvBtn.tintColor = kronosSidebarText();
    [newConvBtn setTitleColor:kronosSidebarText() forState:UIControlStateNormal];
    newConvBtn.titleLabel.font = [UIFont systemFontOfSize:14.5 weight:UIFontWeightSemibold];
    newConvBtn.contentHorizontalAlignment = UIControlContentHorizontalAlignmentLeading;
    newConvBtn.backgroundColor = kronosSidebarPanel();
    newConvBtn.layer.cornerRadius = 11.0;
    newConvBtn.layer.borderWidth = 1.0;
    newConvBtn.layer.borderColor = [UIColor colorWithWhite:1.0 alpha:0.09].CGColor;
    [newConvBtn addTarget:self action:@selector(newConversationFromSidebar) forControlEvents:UIControlEventTouchUpInside];
    [self.sidebarPanel addSubview:newConvBtn];

    UIView *searchBox = [[UIView alloc] init];
    searchBox.translatesAutoresizingMaskIntoConstraints = NO;
    searchBox.backgroundColor = kronosSidebarPanel();
    searchBox.layer.cornerRadius = 9.0;
    [self.sidebarPanel addSubview:searchBox];

    UIImageView *mag = [[UIImageView alloc] initWithImage:[UIImage systemImageNamed:@"magnifyingglass"]];
    mag.translatesAutoresizingMaskIntoConstraints = NO;
    mag.tintColor = kronosSidebarMuted();
    [searchBox addSubview:mag];

    self.sidebarSearchField = [[UITextField alloc] init];
    self.sidebarSearchField.translatesAutoresizingMaskIntoConstraints = NO;
    self.sidebarSearchField.textColor = kronosSidebarText();
    self.sidebarSearchField.font = [UIFont systemFontOfSize:13.5 weight:UIFontWeightRegular];
    self.sidebarSearchField.attributedPlaceholder = [[NSAttributedString alloc] initWithString:@"Search conversations..."
                                                                                    attributes:@{NSForegroundColorAttributeName: kronosSidebarMuted()}];
    self.sidebarSearchField.clearButtonMode = UITextFieldViewModeWhileEditing;
    self.sidebarSearchField.delegate = self;
    [self.sidebarSearchField addTarget:self action:@selector(searchFieldChanged:) forControlEvents:UIControlEventEditingChanged];
    [searchBox addSubview:self.sidebarSearchField];

    UIButton *filesRow = [UIButton buttonWithType:UIButtonTypeSystem];
    filesRow.translatesAutoresizingMaskIntoConstraints = NO;
    [filesRow setTitle:@"  Workspace files" forState:UIControlStateNormal];
    [filesRow setImage:[UIImage systemImageNamed:@"folder"] forState:UIControlStateNormal];
    filesRow.tintColor = kronosAccentColor();
    [filesRow setTitleColor:kronosSidebarText() forState:UIControlStateNormal];
    filesRow.titleLabel.font = [UIFont systemFontOfSize:14.0 weight:UIFontWeightMedium];
    filesRow.contentHorizontalAlignment = UIControlContentHorizontalAlignmentLeading;
    [filesRow addTarget:self action:@selector(openWorkspaceFilesTapped) forControlEvents:UIControlEventTouchUpInside];
    [self.sidebarPanel addSubview:filesRow];

    self.workspaceCountBadge = [[UILabel alloc] init];
    self.workspaceCountBadge.translatesAutoresizingMaskIntoConstraints = NO;
    self.workspaceCountBadge.text = @"0";
    self.workspaceCountBadge.textColor = kronosSidebarMuted();
    self.workspaceCountBadge.font = [UIFont monospacedSystemFontOfSize:11.5 weight:UIFontWeightSemibold];
    [self.sidebarPanel addSubview:self.workspaceCountBadge];

    UILabel *sectionHdr = [[UILabel alloc] init];
    sectionHdr.translatesAutoresizingMaskIntoConstraints = NO;
    sectionHdr.text = @"RECENT CONVERSATIONS";
    sectionHdr.textColor = kronosSidebarMuted();
    sectionHdr.font = [UIFont systemFontOfSize:11.0 weight:UIFontWeightSemibold];
    [self.sidebarPanel addSubview:sectionHdr];

    self.conversationsTable = [[UITableView alloc] initWithFrame:CGRectZero style:UITableViewStylePlain];
    self.conversationsTable.translatesAutoresizingMaskIntoConstraints = NO;
    self.conversationsTable.backgroundColor = [UIColor clearColor];
    self.conversationsTable.separatorStyle = UITableViewCellSeparatorStyleNone;
    self.conversationsTable.dataSource = self;
    self.conversationsTable.delegate = self;
    [self.sidebarPanel addSubview:self.conversationsTable];

    UIControl *settingsFooter = [[UIControl alloc] init];
    settingsFooter.translatesAutoresizingMaskIntoConstraints = NO;
    settingsFooter.backgroundColor = kronosSidebarPanel();
    settingsFooter.layer.cornerRadius = 12.0;
    settingsFooter.layer.borderWidth = 1.0;
    settingsFooter.layer.borderColor = [UIColor colorWithWhite:1.0 alpha:0.08].CGColor;
    [settingsFooter addTarget:self action:@selector(openSettingsFromSidebar) forControlEvents:UIControlEventTouchUpInside];
    [self.sidebarPanel addSubview:settingsFooter];

    UIImageView *footerGear = [[UIImageView alloc] initWithImage:[UIImage systemImageNamed:@"gearshape.fill"]];
    footerGear.translatesAutoresizingMaskIntoConstraints = NO;
    footerGear.tintColor = kronosAccentColor();
    [settingsFooter addSubview:footerGear];

    UILabel *footerTitle = [[UILabel alloc] init];
    footerTitle.translatesAutoresizingMaskIntoConstraints = NO;
    footerTitle.text = @"Settings & Device Context";
    footerTitle.textColor = kronosSidebarText();
    footerTitle.font = [UIFont systemFontOfSize:13.5 weight:UIFontWeightSemibold];
    [settingsFooter addSubview:footerTitle];

    self.sidebarSettingsSubtitle = [[UILabel alloc] init];
    self.sidebarSettingsSubtitle.translatesAutoresizingMaskIntoConstraints = NO;
    self.sidebarSettingsSubtitle.text = @"Device Context On · Phone Use On";
    self.sidebarSettingsSubtitle.textColor = kronosSidebarMuted();
    self.sidebarSettingsSubtitle.font = [UIFont systemFontOfSize:11.0 weight:UIFontWeightRegular];
    [settingsFooter addSubview:self.sidebarSettingsSubtitle];

    UIImageView *footerChev = [[UIImageView alloc] initWithImage:[UIImage systemImageNamed:@"chevron.right"]];
    footerChev.translatesAutoresizingMaskIntoConstraints = NO;
    footerChev.tintColor = kronosSidebarMuted();
    [settingsFooter addSubview:footerChev];

    UILayoutGuide *safe = self.view.safeAreaLayoutGuide;
    self.sidebarLeadingConstraint = [self.sidebarPanel.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor constant:-kSidebarWidth];

    [NSLayoutConstraint activateConstraints:@[
        [self.sidebarBackdrop.topAnchor constraintEqualToAnchor:self.view.topAnchor],
        [self.sidebarBackdrop.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor],
        [self.sidebarBackdrop.trailingAnchor constraintEqualToAnchor:self.view.trailingAnchor],
        [self.sidebarBackdrop.bottomAnchor constraintEqualToAnchor:self.view.bottomAnchor],

        self.sidebarLeadingConstraint,
        [self.sidebarPanel.topAnchor constraintEqualToAnchor:self.view.topAnchor],
        [self.sidebarPanel.bottomAnchor constraintEqualToAnchor:self.view.bottomAnchor],
        [self.sidebarPanel.widthAnchor constraintEqualToConstant:kSidebarWidth],

        [brandLbl.topAnchor constraintEqualToAnchor:safe.topAnchor constant:12.0],
        [brandLbl.leadingAnchor constraintEqualToAnchor:self.sidebarPanel.leadingAnchor constant:16.0],

        [closeBtn.centerYAnchor constraintEqualToAnchor:brandLbl.centerYAnchor],
        [closeBtn.trailingAnchor constraintEqualToAnchor:self.sidebarPanel.trailingAnchor constant:-10.0],
        [closeBtn.widthAnchor constraintEqualToConstant:40.0],
        [closeBtn.heightAnchor constraintEqualToConstant:40.0],

        [newConvBtn.topAnchor constraintEqualToAnchor:brandLbl.bottomAnchor constant:14.0],
        [newConvBtn.leadingAnchor constraintEqualToAnchor:self.sidebarPanel.leadingAnchor constant:14.0],
        [newConvBtn.trailingAnchor constraintEqualToAnchor:self.sidebarPanel.trailingAnchor constant:-14.0],
        [newConvBtn.heightAnchor constraintEqualToConstant:42.0],

        [searchBox.topAnchor constraintEqualToAnchor:newConvBtn.bottomAnchor constant:10.0],
        [searchBox.leadingAnchor constraintEqualToAnchor:newConvBtn.leadingAnchor],
        [searchBox.trailingAnchor constraintEqualToAnchor:newConvBtn.trailingAnchor],
        [searchBox.heightAnchor constraintEqualToConstant:36.0],

        [mag.leadingAnchor constraintEqualToAnchor:searchBox.leadingAnchor constant:10.0],
        [mag.centerYAnchor constraintEqualToAnchor:searchBox.centerYAnchor],
        [mag.widthAnchor constraintEqualToConstant:15.0],
        [mag.heightAnchor constraintEqualToConstant:15.0],

        [self.sidebarSearchField.leadingAnchor constraintEqualToAnchor:mag.trailingAnchor constant:7.0],
        [self.sidebarSearchField.trailingAnchor constraintEqualToAnchor:searchBox.trailingAnchor constant:-8.0],
        [self.sidebarSearchField.centerYAnchor constraintEqualToAnchor:searchBox.centerYAnchor],

        [filesRow.topAnchor constraintEqualToAnchor:searchBox.bottomAnchor constant:10.0],
        [filesRow.leadingAnchor constraintEqualToAnchor:self.sidebarPanel.leadingAnchor constant:16.0],
        [filesRow.trailingAnchor constraintEqualToAnchor:self.workspaceCountBadge.leadingAnchor constant:-8.0],
        [filesRow.heightAnchor constraintEqualToConstant:36.0],

        [self.workspaceCountBadge.centerYAnchor constraintEqualToAnchor:filesRow.centerYAnchor],
        [self.workspaceCountBadge.trailingAnchor constraintEqualToAnchor:self.sidebarPanel.trailingAnchor constant:-18.0],

        [sectionHdr.topAnchor constraintEqualToAnchor:filesRow.bottomAnchor constant:12.0],
        [sectionHdr.leadingAnchor constraintEqualToAnchor:self.sidebarPanel.leadingAnchor constant:16.0],

        [self.conversationsTable.topAnchor constraintEqualToAnchor:sectionHdr.bottomAnchor constant:6.0],
        [self.conversationsTable.leadingAnchor constraintEqualToAnchor:self.sidebarPanel.leadingAnchor constant:8.0],
        [self.conversationsTable.trailingAnchor constraintEqualToAnchor:self.sidebarPanel.trailingAnchor constant:-8.0],
        [self.conversationsTable.bottomAnchor constraintEqualToAnchor:settingsFooter.topAnchor constant:-8.0],

        [settingsFooter.leadingAnchor constraintEqualToAnchor:self.sidebarPanel.leadingAnchor constant:12.0],
        [settingsFooter.trailingAnchor constraintEqualToAnchor:self.sidebarPanel.trailingAnchor constant:-12.0],
        [settingsFooter.bottomAnchor constraintEqualToAnchor:safe.bottomAnchor constant:-10.0],
        [settingsFooter.heightAnchor constraintEqualToConstant:52.0],

        [footerGear.leadingAnchor constraintEqualToAnchor:settingsFooter.leadingAnchor constant:12.0],
        [footerGear.centerYAnchor constraintEqualToAnchor:settingsFooter.centerYAnchor],
        [footerGear.widthAnchor constraintEqualToConstant:20.0],
        [footerGear.heightAnchor constraintEqualToConstant:20.0],

        [footerTitle.topAnchor constraintEqualToAnchor:settingsFooter.topAnchor constant:9.0],
        [footerTitle.leadingAnchor constraintEqualToAnchor:footerGear.trailingAnchor constant:10.0],
        [footerTitle.trailingAnchor constraintEqualToAnchor:footerChev.leadingAnchor constant:-6.0],

        [self.sidebarSettingsSubtitle.topAnchor constraintEqualToAnchor:footerTitle.bottomAnchor constant:2.0],
        [self.sidebarSettingsSubtitle.leadingAnchor constraintEqualToAnchor:footerTitle.leadingAnchor],
        [self.sidebarSettingsSubtitle.trailingAnchor constraintEqualToAnchor:footerTitle.trailingAnchor],

        [footerChev.trailingAnchor constraintEqualToAnchor:settingsFooter.trailingAnchor constant:-12.0],
        [footerChev.centerYAnchor constraintEqualToAnchor:settingsFooter.centerYAnchor],
        [footerChev.widthAnchor constraintEqualToConstant:11.0],
        [footerChev.heightAnchor constraintEqualToConstant:14.0],
    ]];
}

- (void)setupEdgeSwipeGestures {
    UIScreenEdgePanGestureRecognizer *edgePan = [[UIScreenEdgePanGestureRecognizer alloc] initWithTarget:self action:@selector(handleLeftEdgePan:)];
    edgePan.edges = UIRectEdgeLeft;
    [self.view addGestureRecognizer:edgePan];
}

- (void)handleLeftEdgePan:(UIScreenEdgePanGestureRecognizer *)g {
    if (g.state == UIGestureRecognizerStateEnded || g.state == UIGestureRecognizerStateRecognized) {
        if (!self.sidebarOpen) [self openSidebarAnimated];
    }
}

#pragma mark - Settings Sheet & Sidebar Management

- (void)refreshSidebarSettingsFooter {
    KronosMobileSettings *s = [KronosMobileSettings sharedSettings];
    NSString *ctxPart = s.developDeviceContext ? @"Device Context On" : @"Device Context Off";
    NSString *phonePart = s.allowPhoneUse ? @"Phone Use On" : @"Phone Use Off";
    self.sidebarSettingsSubtitle.text = [NSString stringWithFormat:@"%@ · %@", ctxPart, phonePart];
}

- (void)openSettingsFromSidebar {
    [self closeSidebarAnimated];
    [self openSettingsTapped];
}

- (void)openSettingsTapped {
    [self dismissKeyboard];
    KronosSettingsViewController *svc = [[KronosSettingsViewController alloc] init];
    svc.delegate = self;
    UINavigationController *nav = [[UINavigationController alloc] initWithRootViewController:svc];
    nav.modalPresentationStyle = UIModalPresentationPageSheet;
    UISheetPresentationController *sheet = nav.sheetPresentationController;
    if (sheet != nil) {
        sheet.detents = @[[UISheetPresentationControllerDetent mediumDetent],
                          [UISheetPresentationControllerDetent largeDetent]];
        sheet.prefersGrabberVisible = YES;
        sheet.preferredCornerRadius = 22.0;
    }
    [self presentViewController:nav animated:YES completion:nil];
}

- (void)settingsViewControllerDidUpdateSettings {
    self.voiceController.voiceOutputEnabled = [KronosMobileSettings sharedSettings].spokenRepliesEnabled;
    if (!self.voiceController.voiceOutputEnabled) {
        [self.voiceController stopSpeaking];
    }
    [self syncTtsButtonAppearance];
    [self refreshSidebarSettingsFooter];
}

- (void)toggleSidebarTapped {
    if (self.sidebarOpen) [self closeSidebarAnimated];
    else [self openSidebarAnimated];
}

- (void)openSidebarAnimated {
    [self dismissKeyboard];
    [self refreshWorkspaceFileCount];
    [self refreshSidebarSettingsFooter];
    [self applyConversationSearchFilter];
    self.sidebarOpen = YES;
    self.sidebarBackdrop.hidden = NO;
    self.sidebarLeadingConstraint.constant = 0.0;
    [UIView animateWithDuration:0.26 delay:0 options:UIViewAnimationOptionCurveEaseOut animations:^{
        self.sidebarBackdrop.alpha = 1.0;
        [self.view layoutIfNeeded];
    } completion:nil];
}

- (void)closeSidebarAnimated {
    [self.sidebarSearchField resignFirstResponder];
    self.sidebarOpen = NO;
    self.sidebarLeadingConstraint.constant = -kSidebarWidth;
    [UIView animateWithDuration:0.22 delay:0 options:UIViewAnimationOptionCurveEaseIn animations:^{
        self.sidebarBackdrop.alpha = 0.0;
        [self.view layoutIfNeeded];
    } completion:^(BOOL finished) {
        (void)finished;
        if (!self.sidebarOpen) self.sidebarBackdrop.hidden = YES;
    }];
}

- (void)searchFieldChanged:(UITextField *)tf {
    (void)tf;
    [self applyConversationSearchFilter];
}

- (void)applyConversationSearchFilter {
    NSString *q = [[self.sidebarSearchField.text ?: @"" stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]] lowercaseString];
    if (q.length == 0) {
        self.filteredConversations = [self.conversations mutableCopy];
    } else {
        NSMutableArray *matched = [NSMutableArray array];
        for (NSMutableDictionary *c in self.conversations) {
            NSString *title = [c[@"title"] ?: @"" lowercaseString];
            if ([title containsString:q]) {
                [matched addObject:c];
            }
        }
        self.filteredConversations = matched;
    }
    [self.conversationsTable reloadData];
}

- (void)newConversationFromSidebar {
    [self closeSidebarAnimated];
    [self newConversationTapped];
}

- (void)newConversationTapped {
    if (self.isGenerating) atomic_store(&_cancelFlag, true);
    [self.voiceController stopSpeaking];
    self.pendingWhatsAppRecipient = nil;
    self.pendingWhatsAppAwaitingRecipient = NO;

    NSArray *curMsgs = self.activeConversation[@"messages"];
    if (curMsgs.count == 0) {
        [self renderActiveConversationTranscript];
        return;
    }
    NSMutableDictionary *fresh = [KronosConversationStore createBlankConversation];
    [self.conversations insertObject:fresh atIndex:0];
    self.activeConversation = fresh;
    [KronosConversationStore saveConversations:self.conversations];
    [self applyConversationSearchFilter];
    [self renderActiveConversationTranscript];

    dispatch_async(self.engineQueue, ^{
        if (self.session != NULL) kronos_session_reset(self.session);
    });
}

- (NSInteger)tableView:(UITableView *)tableView numberOfRowsInSection:(NSInteger)section {
    (void)tableView;
    (void)section;
    return (NSInteger)self.filteredConversations.count;
}

- (UITableViewCell *)tableView:(UITableView *)tableView cellForRowAtIndexPath:(NSIndexPath *)indexPath {
    UITableViewCell *cell = [tableView dequeueReusableCellWithIdentifier:@"ConvCell"];
    if (cell == nil) {
        cell = [[UITableViewCell alloc] initWithStyle:UITableViewCellStyleSubtitle reuseIdentifier:@"ConvCell"];
        cell.backgroundColor = [UIColor clearColor];
        cell.layer.cornerRadius = 10.0;
        cell.clipsToBounds = YES;
    }
    NSDictionary *conv = self.filteredConversations[(NSUInteger)indexPath.row];
    BOOL isActive = [conv[@"id"] isEqual:self.activeConversation[@"id"]];
    cell.contentView.backgroundColor = isActive ? kronosSidebarActive() : [UIColor clearColor];
    cell.textLabel.text = conv[@"title"] ?: @"New conversation";
    cell.textLabel.textColor = kronosSidebarText();
    cell.textLabel.font = [UIFont systemFontOfSize:14.0 weight:isActive ? UIFontWeightSemibold : UIFontWeightRegular];
    NSTimeInterval ts = [conv[@"updatedAt"] doubleValue];
    cell.detailTextLabel.text = kronosRelativeDateString(ts);
    cell.detailTextLabel.textColor = kronosSidebarMuted();
    cell.detailTextLabel.font = [UIFont systemFontOfSize:11.5 weight:UIFontWeightRegular];
    return cell;
}

- (void)tableView:(UITableView *)tableView didSelectRowAtIndexPath:(NSIndexPath *)indexPath {
    [tableView deselectRowAtIndexPath:indexPath animated:YES];
    if (self.isGenerating) atomic_store(&_cancelFlag, true);
    self.pendingWhatsAppRecipient = nil;
    self.pendingWhatsAppAwaitingRecipient = NO;
    NSMutableDictionary *picked = self.filteredConversations[(NSUInteger)indexPath.row];
    self.activeConversation = picked;
    [self closeSidebarAnimated];
    [self renderActiveConversationTranscript];
    dispatch_async(self.engineQueue, ^{
        if (self.session != NULL) kronos_session_reset(self.session);
    });
}

- (UISwipeActionsConfiguration *)tableView:(UITableView *)tableView trailingSwipeActionsConfigurationForRowAtIndexPath:(NSIndexPath *)indexPath {
    (void)tableView;
    UIContextualAction *del = [UIContextualAction contextualActionWithStyle:UIContextualActionStyleDestructive
                                                                      title:@"Delete"
                                                                    handler:^(UIContextualAction *action, UIView *sourceView, void (^completionHandler)(BOOL)) {
        (void)action;
        (void)sourceView;
        NSMutableDictionary *target = self.filteredConversations[(NSUInteger)indexPath.row];
        [self.conversations removeObject:target];
        if (self.conversations.count == 0) {
            [self.conversations addObject:[KronosConversationStore createBlankConversation]];
        }
        if ([target[@"id"] isEqual:self.activeConversation[@"id"]]) {
            self.activeConversation = self.conversations.firstObject;
            [self renderActiveConversationTranscript];
            dispatch_async(self.engineQueue, ^{
                if (self.session != NULL) kronos_session_reset(self.session);
            });
        }
        [KronosConversationStore saveConversations:self.conversations];
        [self applyConversationSearchFilter];
        completionHandler(YES);
    }];
    return [UISwipeActionsConfiguration configurationWithActions:@[del]];
}

#pragma mark - Welcome Screen & Transcript Rendering

- (void)renderActiveConversationTranscript {
    for (UIView *sub in [self.messagesStack.arrangedSubviews copy]) {
        [self.messagesStack removeArrangedSubview:sub];
        [sub removeFromSuperview];
    }
    self.welcomeView = nil;

    NSArray<NSDictionary *> *msgs = self.activeConversation[@"messages"];
    NSString *convTitle = self.activeConversation[@"title"] ?: @"Kronos";
    self.headerTitleLabel.text = ([convTitle isEqualToString:@"New conversation"] || msgs.count == 0) ? @"Kronos" : convTitle;

    if (msgs.count == 0) {
        [self buildWelcomeCard];
        return;
    }

    for (NSDictionary *m in msgs) {
        NSString *role = m[@"role"] ?: @"";
        NSString *content = m[@"displayText"] ?: (m[@"content"] ?: @"");
        NSArray<NSString *> *attNames = m[@"attachments"];
        NSString *actionBadge = m[@"actionBadge"];
        NSString *previewFile = m[@"previewFile"];
        NSString *actionType = m[@"actionType"];
        NSString *draftURLStr = m[@"draftURL"];
        NSString *draftPostText = m[@"draftPostText"];
        NSString *previewImageFile = m[@"previewImageFile"];

        if ([role isEqualToString:@"user"]) {
            [self appendUserBubbleWithText:content attachmentNames:attNames thumbnails:nil];
        } else {
            UILabel *body = nil;
            UIStackView *extra = nil;
            [self appendKronosBubbleOutBody:&body outExtraStack:&extra];
            body.attributedText = kronosFormatMessageText(content, NO);

            if (previewImageFile.length > 0) {
                NSString *imgPath = [kronosWorkspaceDirectory() stringByAppendingPathComponent:previewImageFile];
                UIImage *savedImg = [UIImage imageWithContentsOfFile:imgPath];
                if (savedImg != nil) {
                    [self addPhotoPreviewCardWithImage:savedImg caption:@"Selected photo from Photo Library" toStack:extra];
                }
            }
            if ([actionType isEqualToString:@"whatsapp"] && draftURLStr.length > 0) {
                self.lastWhatsAppDraftURL = [NSURL URLWithString:draftURLStr];
                [self addWhatsAppOpenDraftButtonToStack:extra];
            } else if ([actionType isEqualToString:@"linkedin"] && draftPostText.length > 0) {
                self.lastLinkedInDraftText = draftPostText;
                if (previewImageFile.length > 0) {
                    NSString *imgPath = [kronosWorkspaceDirectory() stringByAppendingPathComponent:previewImageFile];
                    self.lastLinkedInDraftPhoto = [UIImage imageWithContentsOfFile:imgPath];
                }
                [self addLinkedInPublishButtonsToStack:extra];
            } else if (actionBadge.length > 0) {
                [self addActionPillWithTitle:actionBadge previewFile:previewFile toStack:extra];
            }
        }
    }
    [self scrollToBottomAnimated:NO];
}

- (void)buildWelcomeCard {
    UIView *wrap = [[UIView alloc] init];
    wrap.translatesAutoresizingMaskIntoConstraints = NO;

    UIStackView *col = [[UIStackView alloc] init];
    col.translatesAutoresizingMaskIntoConstraints = NO;
    col.axis = UILayoutConstraintAxisVertical;
    col.alignment = UIStackViewAlignmentFill;
    col.spacing = 12.0;
    [wrap addSubview:col];

    UIView *hero = [[UIView alloc] init];
    hero.translatesAutoresizingMaskIntoConstraints = NO;

    UIView *badge = [[UIView alloc] init];
    badge.translatesAutoresizingMaskIntoConstraints = NO;
    badge.backgroundColor = [kronosAccentColor() colorWithAlphaComponent:0.14];
    badge.layer.cornerRadius = 26.0;
    [hero addSubview:badge];

    UIImageView *spark = [[UIImageView alloc] initWithImage:[UIImage systemImageNamed:@"sparkles"]];
    spark.translatesAutoresizingMaskIntoConstraints = NO;
    spark.tintColor = kronosAccentColor();
    [badge addSubview:spark];

    UILabel *hTitle = [[UILabel alloc] init];
    hTitle.translatesAutoresizingMaskIntoConstraints = NO;
    hTitle.text = @"How can I help you today?";
    hTitle.textColor = kronosTextColor();
    hTitle.font = [UIFont systemFontOfSize:22.0 weight:UIFontWeightBold];
    hTitle.textAlignment = NSTextAlignmentCenter;
    [hero addSubview:hTitle];

    UILabel *hSub = [[UILabel alloc] init];
    hSub.translatesAutoresizingMaskIntoConstraints = NO;
    hSub.text = @"Talk by voice, attach photos and files, draft WhatsApp messages, prepare LinkedIn posts, or build pages locally on your iPhone.";
    hSub.textColor = kronosMutedColor();
    hSub.font = [UIFont systemFontOfSize:14.5 weight:UIFontWeightRegular];
    hSub.textAlignment = NSTextAlignmentCenter;
    hSub.numberOfLines = 0;
    [hero addSubview:hSub];

    [NSLayoutConstraint activateConstraints:@[
        [badge.topAnchor constraintEqualToAnchor:hero.topAnchor constant:20.0],
        [badge.centerXAnchor constraintEqualToAnchor:hero.centerXAnchor],
        [badge.widthAnchor constraintEqualToConstant:52.0],
        [badge.heightAnchor constraintEqualToConstant:52.0],

        [spark.centerXAnchor constraintEqualToAnchor:badge.centerXAnchor],
        [spark.centerYAnchor constraintEqualToAnchor:badge.centerYAnchor],
        [spark.widthAnchor constraintEqualToConstant:24.0],
        [spark.heightAnchor constraintEqualToConstant:24.0],

        [hTitle.topAnchor constraintEqualToAnchor:badge.bottomAnchor constant:14.0],
        [hTitle.leadingAnchor constraintEqualToAnchor:hero.leadingAnchor constant:8.0],
        [hTitle.trailingAnchor constraintEqualToAnchor:hero.trailingAnchor constant:-8.0],

        [hSub.topAnchor constraintEqualToAnchor:hTitle.bottomAnchor constant:6.0],
        [hSub.leadingAnchor constraintEqualToAnchor:hero.leadingAnchor constant:12.0],
        [hSub.trailingAnchor constraintEqualToAnchor:hero.trailingAnchor constant:-12.0],
        [hSub.bottomAnchor constraintEqualToAnchor:hero.bottomAnchor constant:-14.0],
    ]];
    [col addArrangedSubview:hero];

    NSArray<NSArray<NSString *> *> *suggestions = @[
        @[@"person.text.rectangle", @"Tell me about Melody Leonard"],
        @[@"message.fill", @"Open WhatsApp and write a message to Dera"],
        @[@"photo.on.rectangle.angled", @"Create a Monday motivation LinkedIn post"],
        @[@"safari", @"Build a personal portfolio webpage"],
    ];

    for (NSUInteger i = 0; i < suggestions.count; i++) {
        NSArray<NSString *> *s = suggestions[i];
        UIButton *card = [UIButton buttonWithType:UIButtonTypeSystem];
        card.translatesAutoresizingMaskIntoConstraints = NO;
        card.tag = (NSInteger)(5000 + i);
        [card setImage:[UIImage systemImageNamed:s[0]] forState:UIControlStateNormal];
        [card setTitle:[NSString stringWithFormat:@"   %@", s[1]] forState:UIControlStateNormal];
        card.tintColor = kronosAccentColor();
        [card setTitleColor:kronosTextColor() forState:UIControlStateNormal];
        card.titleLabel.font = [UIFont systemFontOfSize:14.5 weight:UIFontWeightMedium];
        card.titleLabel.lineBreakMode = NSLineBreakByTruncatingTail;
        card.contentHorizontalAlignment = UIControlContentHorizontalAlignmentLeading;
        [card.heightAnchor constraintEqualToConstant:50.0].active = YES;
        [card addTarget:self action:@selector(suggestionCardTapped:) forControlEvents:UIControlEventTouchUpInside];

        UIView *padWrap = [[UIView alloc] init];
        padWrap.translatesAutoresizingMaskIntoConstraints = NO;
        padWrap.backgroundColor = kronosPanelColor();
        padWrap.layer.cornerRadius = 14.0;
        padWrap.layer.borderWidth = 1.0;
        padWrap.layer.borderColor = kronosLineColor().CGColor;
        [padWrap addSubview:card];
        [NSLayoutConstraint activateConstraints:@[
            [card.topAnchor constraintEqualToAnchor:padWrap.topAnchor],
            [card.bottomAnchor constraintEqualToAnchor:padWrap.bottomAnchor],
            [card.leadingAnchor constraintEqualToAnchor:padWrap.leadingAnchor constant:14.0],
            [card.trailingAnchor constraintEqualToAnchor:padWrap.trailingAnchor constant:-14.0],
        ]];
        [col addArrangedSubview:padWrap];
    }

    [NSLayoutConstraint activateConstraints:@[
        [col.topAnchor constraintEqualToAnchor:wrap.topAnchor],
        [col.leadingAnchor constraintEqualToAnchor:wrap.leadingAnchor],
        [col.trailingAnchor constraintEqualToAnchor:wrap.trailingAnchor],
        [col.bottomAnchor constraintEqualToAnchor:wrap.bottomAnchor],
    ]];

    self.welcomeView = wrap;
    [self.messagesStack addArrangedSubview:wrap];
}

- (void)suggestionCardTapped:(UIButton *)sender {
    NSArray<NSString *> *prompts = @[
        @"Tell me about Melody Leonard and his engineering background",
        @"Open WhatsApp and write a message to Dera",
        @"Create a LinkedIn post with the last picture I took today and talk about Monday motivation",
        @"Create index.html for Melody Leonard's software engineering portfolio",
    ];
    NSUInteger idx = (NSUInteger)(sender.tag - 5000);
    if (idx < prompts.count && !self.isGenerating) {
        self.lastTurnWasSpoken = NO;
        [self submitUserPrompt:prompts[idx]];
    }
}

#pragma mark - Attachments & Clipboard Paste

- (void)attachTapped {
    [self dismissKeyboard];
    UIAlertController *sheet = [UIAlertController alertControllerWithTitle:nil
                                                                   message:nil
                                                            preferredStyle:UIAlertControllerStyleActionSheet];
    [sheet addAction:[UIAlertAction actionWithTitle:@"Photo Library" style:UIAlertActionStyleDefault handler:^(UIAlertAction *a) {
        (void)a;
        PHPickerConfiguration *cfg = [[PHPickerConfiguration alloc] init];
        cfg.selectionLimit = 4;
        cfg.filter = [PHPickerFilter imagesFilter];
        PHPickerViewController *picker = [[PHPickerViewController alloc] initWithConfiguration:cfg];
        picker.delegate = self;
        [self presentViewController:picker animated:YES completion:nil];
    }]];

    if ([UIImagePickerController isSourceTypeAvailable:UIImagePickerControllerSourceTypeCamera]) {
        [sheet addAction:[UIAlertAction actionWithTitle:@"Take Photo" style:UIAlertActionStyleDefault handler:^(UIAlertAction *a) {
            (void)a;
            UIImagePickerController *cam = [[UIImagePickerController alloc] init];
            cam.sourceType = UIImagePickerControllerSourceTypeCamera;
            cam.delegate = self;
            [self presentViewController:cam animated:YES completion:nil];
        }]];
    }

    [sheet addAction:[UIAlertAction actionWithTitle:@"Choose File" style:UIAlertActionStyleDefault handler:^(UIAlertAction *a) {
        (void)a;
        UIDocumentPickerViewController *doc = [[UIDocumentPickerViewController alloc] initForOpeningContentTypes:@[UTTypeItem] asCopy:YES];
        doc.allowsMultipleSelection = YES;
        doc.delegate = self;
        [self presentViewController:doc animated:YES completion:nil];
    }]];

    UIPasteboard *pb = [UIPasteboard generalPasteboard];
    if (pb.hasImages || pb.hasURLs || pb.hasStrings || pb.items.count > 0) {
        [sheet addAction:[UIAlertAction actionWithTitle:@"Paste from Clipboard" style:UIAlertActionStyleDefault handler:^(UIAlertAction *a) {
            (void)a;
            if (pb.hasImages && pb.image != nil) {
                [self inputTextViewDidPasteImage:pb.image];
            } else if (pb.hasStrings && pb.string.length > 0) {
                self.inputTextView.text = [self.inputTextView.text stringByAppendingString:pb.string];
                [self textViewDidChange:self.inputTextView];
            } else {
                [self.inputTextView paste:nil];
            }
        }]];
    }

    [sheet addAction:[UIAlertAction actionWithTitle:@"Cancel" style:UIAlertActionStyleCancel handler:nil]];
    [self presentViewController:sheet animated:YES completion:nil];
}

- (void)inputTextViewDidPasteImage:(UIImage *)image {
    [self enqueueImageAttachment:image suggestedName:[NSString stringWithFormat:@"pasted_image_%lu.png", (unsigned long)(self.pendingAttachments.count + 1)]];
}

- (void)inputTextViewDidPasteFileData:(NSData *)data filename:(NSString *)filename mime:(NSString *)mime {
    [self enqueueFileAttachmentData:data filename:filename mime:mime];
}

- (void)enqueueImageAttachment:(UIImage *)img suggestedName:(NSString *)name {
    self.headerSubtitleLabel.text = @"Analyzing image...";
    NSUInteger slot = self.pendingAttachments.count + 1;
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        KronosAttachmentItem *item = kronosBuildAttachmentFromImage(img, name, slot);
        dispatch_async(dispatch_get_main_queue(), ^{
            self.headerSubtitleLabel.text = @"";
            if (item != nil) {
                [self.pendingAttachments addObject:item];
                [self rebuildPendingAttachmentsStrip];
                [self refreshWorkspaceFileCount];
                [self updateComposerControls];
            }
        });
    });
}

- (void)enqueueFileAttachmentData:(NSData *)data filename:(NSString *)name mime:(NSString *)mime {
    self.headerSubtitleLabel.text = @"Reading file...";
    NSUInteger slot = self.pendingAttachments.count + 1;
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        KronosAttachmentItem *item = kronosBuildAttachmentFromData(data, name, mime, slot);
        dispatch_async(dispatch_get_main_queue(), ^{
            self.headerSubtitleLabel.text = @"";
            if (item != nil) {
                [self.pendingAttachments addObject:item];
                [self rebuildPendingAttachmentsStrip];
                [self refreshWorkspaceFileCount];
                [self updateComposerControls];
            }
        });
    });
}

- (void)picker:(PHPickerViewController *)picker didFinishPicking:(NSArray<PHPickerResult *> *)results {
    [picker dismissViewControllerAnimated:YES completion:nil];
    for (PHPickerResult *res in results) {
        if ([res.itemProvider canLoadObjectOfClass:[UIImage class]]) {
            NSString *suggested = res.itemProvider.suggestedName ?: @"photo.png";
            [res.itemProvider loadObjectOfClass:[UIImage class] completionHandler:^(id<NSItemProviderReading> obj, NSError *err) {
                (void)err;
                if ([obj isKindOfClass:[UIImage class]]) {
                    dispatch_async(dispatch_get_main_queue(), ^{
                        [self enqueueImageAttachment:(UIImage *)obj suggestedName:suggested];
                    });
                }
            }];
        }
    }
}

- (void)imagePickerController:(UIImagePickerController *)picker didFinishPickingMediaWithInfo:(NSDictionary<UIImagePickerControllerInfoKey,id> *)info {
    [picker dismissViewControllerAnimated:YES completion:nil];
    UIImage *img = info[UIImagePickerControllerOriginalImage];
    if (img != nil) {
        [self enqueueImageAttachment:img suggestedName:@"camera_photo.png"];
    }
}

- (void)documentPicker:(UIDocumentPickerViewController *)controller didPickDocumentsAtURLs:(NSArray<NSURL *> *)urls {
    (void)controller;
    for (NSURL *url in urls) {
        BOOL sec = [url startAccessingSecurityScopedResource];
        NSData *data = [NSData dataWithContentsOfURL:url];
        if (sec) [url stopAccessingSecurityScopedResource];
        if (data.length > 0) {
            [self enqueueFileAttachmentData:data filename:url.lastPathComponent mime:@""];
        }
    }
}

- (void)rebuildPendingAttachmentsStrip {
    for (UIView *sub in [self.attachmentsStack.arrangedSubviews copy]) {
        [self.attachmentsStack removeArrangedSubview:sub];
        [sub removeFromSuperview];
    }
    BOOL hasItems = (self.pendingAttachments.count > 0);
    self.attachmentsStrip.hidden = !hasItems;
    self.attachmentsHeightConstraint.constant = hasItems ? 58.0 : 0.0;

    for (NSUInteger i = 0; i < self.pendingAttachments.count; i++) {
        KronosAttachmentItem *att = self.pendingAttachments[i];
        UIView *chip = [[UIView alloc] init];
        chip.translatesAutoresizingMaskIntoConstraints = NO;
        chip.backgroundColor = kronosPanelColor();
        chip.layer.cornerRadius = 12.0;
        chip.layer.borderWidth = 1.0;
        chip.layer.borderColor = kronosLineColor().CGColor;

        UIImageView *thumb = [[UIImageView alloc] init];
        thumb.translatesAutoresizingMaskIntoConstraints = NO;
        thumb.layer.cornerRadius = 8.0;
        thumb.clipsToBounds = YES;
        thumb.contentMode = UIViewContentModeScaleAspectFill;
        if (att.isImage && att.thumbnail != nil) {
            thumb.image = att.thumbnail;
        } else {
            thumb.image = [UIImage systemImageNamed:@"doc.text.fill"];
            thumb.tintColor = kronosAccentColor();
            thumb.contentMode = UIViewContentModeScaleAspectFit;
        }
        [chip addSubview:thumb];

        UILabel *nameLbl = [[UILabel alloc] init];
        nameLbl.translatesAutoresizingMaskIntoConstraints = NO;
        nameLbl.text = att.filename;
        nameLbl.textColor = kronosTextColor();
        nameLbl.font = [UIFont systemFontOfSize:12.5 weight:UIFontWeightSemibold];
        [chip addSubview:nameLbl];

        UILabel *subLbl = [[UILabel alloc] init];
        subLbl.translatesAutoresizingMaskIntoConstraints = NO;
        subLbl.text = att.summaryBadge;
        subLbl.textColor = kronosMutedColor();
        subLbl.font = [UIFont systemFontOfSize:11.0 weight:UIFontWeightRegular];
        [chip addSubview:subLbl];

        UIButton *rm = [UIButton buttonWithType:UIButtonTypeSystem];
        rm.translatesAutoresizingMaskIntoConstraints = NO;
        rm.tag = (NSInteger)(7000 + i);
        [rm setImage:[UIImage systemImageNamed:@"xmark.circle.fill"] forState:UIControlStateNormal];
        rm.tintColor = kronosMutedColor();
        [rm addTarget:self action:@selector(removeAttachmentTapped:) forControlEvents:UIControlEventTouchUpInside];
        [chip addSubview:rm];

        [NSLayoutConstraint activateConstraints:@[
            [chip.heightAnchor constraintEqualToConstant:54.0],
            [thumb.leadingAnchor constraintEqualToAnchor:chip.leadingAnchor constant:7.0],
            [thumb.centerYAnchor constraintEqualToAnchor:chip.centerYAnchor],
            [thumb.widthAnchor constraintEqualToConstant:38.0],
            [thumb.heightAnchor constraintEqualToConstant:38.0],

            [nameLbl.topAnchor constraintEqualToAnchor:chip.topAnchor constant:9.0],
            [nameLbl.leadingAnchor constraintEqualToAnchor:thumb.trailingAnchor constant:8.0],
            [nameLbl.trailingAnchor constraintEqualToAnchor:rm.leadingAnchor constant:-6.0],
            [nameLbl.widthAnchor constraintLessThanOrEqualToConstant:130.0],

            [subLbl.topAnchor constraintEqualToAnchor:nameLbl.bottomAnchor constant:2.0],
            [subLbl.leadingAnchor constraintEqualToAnchor:nameLbl.leadingAnchor],
            [subLbl.trailingAnchor constraintEqualToAnchor:nameLbl.trailingAnchor],

            [rm.trailingAnchor constraintEqualToAnchor:chip.trailingAnchor constant:-6.0],
            [rm.centerYAnchor constraintEqualToAnchor:chip.centerYAnchor],
            [rm.widthAnchor constraintEqualToConstant:26.0],
            [rm.heightAnchor constraintEqualToConstant:26.0],
        ]];
        [self.attachmentsStack addArrangedSubview:chip];
    }
}

- (void)removeAttachmentTapped:(UIButton *)sender {
    NSUInteger idx = (NSUInteger)(sender.tag - 7000);
    if (idx < self.pendingAttachments.count) {
        [self.pendingAttachments removeObjectAtIndex:idx];
        [self rebuildPendingAttachmentsStrip];
        [self updateComposerControls];
    }
}

#pragma mark - Voice Input & Spoken Output

- (void)micTapped {
    [self dismissKeyboard];
    [self.voiceController toggleListening];
}

- (void)syncTtsButtonAppearance {
    UIImageSymbolConfiguration *spkCfg = [UIImageSymbolConfiguration configurationWithPointSize:15.0 weight:UIImageSymbolWeightMedium];
    NSString *icon = self.voiceController.voiceOutputEnabled ? @"speaker.wave.2.fill" : @"speaker.slash";
    [self.ttsToggleButton setImage:[UIImage systemImageNamed:icon withConfiguration:spkCfg] forState:UIControlStateNormal];
    self.ttsToggleButton.tintColor = self.voiceController.voiceOutputEnabled ? kronosAccentColor() : kronosMutedColor();
}

- (void)ttsToggleTapped {
    self.voiceController.voiceOutputEnabled = !self.voiceController.voiceOutputEnabled;
    [KronosMobileSettings sharedSettings].spokenRepliesEnabled = self.voiceController.voiceOutputEnabled;
    [[KronosMobileSettings sharedSettings] save];
    if (!self.voiceController.voiceOutputEnabled) {
        [self.voiceController stopSpeaking];
    }
    [self syncTtsButtonAppearance];
}

- (void)voiceControllerDidChangeListening:(BOOL)isListening statusText:(NSString *)status {
    UIImageSymbolConfiguration *iconCfg = [UIImageSymbolConfiguration configurationWithPointSize:18.0 weight:UIImageSymbolWeightMedium];
    if (isListening) {
        [self.micButton setImage:[UIImage systemImageNamed:@"waveform" withConfiguration:iconCfg] forState:UIControlStateNormal];
        self.micButton.tintColor = [UIColor whiteColor];
        self.micButton.backgroundColor = [UIColor colorWithRed:0.88 green:0.26 blue:0.28 alpha:1.0];
        self.headerSubtitleLabel.text = status ?: @"Listening...";
    } else {
        [self.micButton setImage:[UIImage systemImageNamed:@"mic.fill" withConfiguration:iconCfg] forState:UIControlStateNormal];
        self.micButton.tintColor = kronosMutedColor();
        self.micButton.backgroundColor = [UIColor clearColor];
        self.headerSubtitleLabel.text = status ?: @"";
    }
}

- (void)voiceControllerDidTranscribePartial:(NSString *)transcript {
    self.inputTextView.text = transcript;
    [self textViewDidChange:self.inputTextView];
}

- (void)voiceControllerDidFinishUtterance:(NSString *)finalTranscript {
    self.inputTextView.text = @"";
    [self textViewDidChange:self.inputTextView];
    if (finalTranscript.length > 0 && !self.isGenerating) {
        self.lastTurnWasSpoken = YES;
        [self submitUserPrompt:finalTranscript];
    }
}

#pragma mark - Engine & Workspace Files

- (void)refreshWorkspaceFileCount {
    NSArray<NSString *> *raw = [[NSFileManager defaultManager] contentsOfDirectoryAtPath:kronosWorkspaceDirectory() error:nil] ?: @[];
    NSUInteger count = 0;
    for (NSString *f in raw) {
        if (![f hasPrefix:@"."]) count++;
    }
    self.workspaceCountBadge.text = [NSString stringWithFormat:@"%lu", (unsigned long)count];
}

- (void)openWorkspaceFilesTapped {
    [self closeSidebarAnimated];
    KronosWorkspaceListViewController *vc = [[KronosWorkspaceListViewController alloc] init];
    UINavigationController *nav = [[UINavigationController alloc] initWithRootViewController:vc];
    [self presentViewController:nav animated:YES completion:nil];
}

- (void)openFilePreviewForFilename:(NSString *)filename {
    NSString *full = [kronosWorkspaceDirectory() stringByAppendingPathComponent:filename];
    if (![[NSFileManager defaultManager] fileExistsAtPath:full]) return;
    KronosFilePreviewViewController *pvc = [[KronosFilePreviewViewController alloc] init];
    pvc.filePath = full;
    UINavigationController *nav = [[UINavigationController alloc] initWithRootViewController:pvc];
    [self presentViewController:nav animated:YES completion:nil];
}

- (void)loadLocalEngineInBackground {
    NSString *bundlePath = [[NSBundle mainBundle] bundlePath];
    NSString *instructDir = [bundlePath stringByAppendingPathComponent:@"model_instruct"];
    NSString *fallbackDir = [bundlePath stringByAppendingPathComponent:@"model_05b"];
    NSString *modelDir = [[NSFileManager defaultManager] fileExistsAtPath:instructDir] ? instructDir : fallbackDir;

    dispatch_async(self.engineQueue, ^{
        KronosMobileSession *sess = kronos_session_create([modelDir UTF8String], "nf4g128", 2048, 0, 1850);
        if (sess != NULL) {
            kronos_session_configure_kv(sess, 8, 2048, 64);
            kronos_session_enable_metal(sess);
            self.session = sess;
        }
        dispatch_async(dispatch_get_main_queue(), ^{
            self.modelReady = (sess != NULL);
            self.headerSubtitleLabel.text = self.modelReady ? @"" : @"Unavailable";
            [self updateComposerControls];
        });
    });
}

#pragma mark - Composer Input & Message Bubbles

- (void)dismissKeyboard {
    [self.view endEditing:YES];
}

- (void)updateComposerControls {
    UIImageSymbolConfiguration *sendCfg = [UIImageSymbolConfiguration configurationWithPointSize:28.0 weight:UIImageSymbolWeightSemibold];
    if (self.isGenerating) {
        self.sendButton.enabled = YES;
        [self.sendButton setImage:[UIImage systemImageNamed:@"stop.circle.fill" withConfiguration:sendCfg] forState:UIControlStateNormal];
        self.sendButton.tintColor = [UIColor colorWithRed:0.86 green:0.25 blue:0.27 alpha:1.0];
    } else {
        NSString *trimmed = [self.inputTextView.text stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
        BOOL hasContent = (trimmed.length > 0 || self.pendingAttachments.count > 0);
        self.sendButton.enabled = hasContent;
        [self.sendButton setImage:[UIImage systemImageNamed:@"arrow.up.circle.fill" withConfiguration:sendCfg] forState:UIControlStateNormal];
        self.sendButton.tintColor = hasContent ? kronosAccentColor() : [kronosMutedColor() colorWithAlphaComponent:0.35];
    }
}

- (void)textViewDidChange:(UITextView *)textView {
    self.placeholderLabel.hidden = (textView.text.length > 0);
    CGFloat maxW = textView.bounds.size.width > 20 ? textView.bounds.size.width : 220.0;
    CGSize fit = [textView sizeThatFits:CGSizeMake(maxW, 130.0)];
    CGFloat clamped = MIN(MAX(fit.height, 38.0), 118.0);
    textView.scrollEnabled = (fit.height > 118.0);
    self.inputHeightConstraint.constant = clamped;
    [self updateComposerControls];
}

- (void)scrollToBottomAnimated:(BOOL)animated {
    [self.view layoutIfNeeded];
    CGFloat y = self.scrollView.contentSize.height - self.scrollView.bounds.size.height + self.scrollView.contentInset.bottom;
    if (y > 0) [self.scrollView setContentOffset:CGPointMake(0, y) animated:animated];
}

- (void)sendOrStopTapped {
    if (self.voiceController.isListening) {
        [self.voiceController stopListeningAndCommit:YES];
        return;
    }
    if (self.isGenerating) {
        atomic_store(&_cancelFlag, true);
        [self.voiceController stopSpeaking];
        return;
    }
    NSString *trimmed = [self.inputTextView.text stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
    if (trimmed.length == 0 && self.pendingAttachments.count == 0) return;
    if (trimmed.length == 0 && self.pendingAttachments.count > 0) {
        trimmed = @"Identify and describe what I attached.";
    }
    self.inputTextView.text = @"";
    [self textViewDidChange:self.inputTextView];
    [self.inputTextView resignFirstResponder];
    self.lastTurnWasSpoken = NO;
    [self submitUserPrompt:trimmed];
}

- (void)appendUserBubbleWithText:(NSString *)text
                 attachmentNames:(NSArray<NSString *> *)attNames
                      thumbnails:(NSArray<UIImage *> *)thumbs {
    UIView *row = [[UIView alloc] init];
    row.translatesAutoresizingMaskIntoConstraints = NO;

    UIView *bubble = [[UIView alloc] init];
    bubble.translatesAutoresizingMaskIntoConstraints = NO;
    bubble.backgroundColor = kronosUserBubbleBg();
    bubble.layer.cornerRadius = 18.0;
    [row addSubview:bubble];

    UIStackView *inner = [[UIStackView alloc] init];
    inner.translatesAutoresizingMaskIntoConstraints = NO;
    inner.axis = UILayoutConstraintAxisVertical;
    inner.spacing = 8.0;
    [bubble addSubview:inner];

    if (thumbs.count > 0) {
        UIStackView *imgRow = [[UIStackView alloc] init];
        imgRow.axis = UILayoutConstraintAxisHorizontal;
        imgRow.spacing = 6.0;
        for (UIImage *im in thumbs) {
            UIImageView *iv = [[UIImageView alloc] initWithImage:im];
            iv.translatesAutoresizingMaskIntoConstraints = NO;
            iv.contentMode = UIViewContentModeScaleAspectFill;
            iv.layer.cornerRadius = 10.0;
            iv.clipsToBounds = YES;
            [iv.widthAnchor constraintEqualToConstant:110.0].active = YES;
            [iv.heightAnchor constraintEqualToConstant:110.0].active = YES;
            [imgRow addArrangedSubview:iv];
        }
        [inner addArrangedSubview:imgRow];
    } else if (attNames.count > 0) {
        for (NSString *fn in attNames) {
            UILabel *pill = [[UILabel alloc] init];
            pill.text = [NSString stringWithFormat:@"Attached: %@", fn];
            pill.textColor = [UIColor colorWithWhite:1.0 alpha:0.9];
            pill.font = [UIFont systemFontOfSize:12.5 weight:UIFontWeightSemibold];
            [inner addArrangedSubview:pill];
        }
    }

    UILabel *lbl = [[UILabel alloc] init];
    lbl.attributedText = kronosFormatMessageText(text, YES);
    lbl.numberOfLines = 0;
    [inner addArrangedSubview:lbl];

    [NSLayoutConstraint activateConstraints:@[
        [bubble.topAnchor constraintEqualToAnchor:row.topAnchor],
        [bubble.bottomAnchor constraintEqualToAnchor:row.bottomAnchor],
        [bubble.trailingAnchor constraintEqualToAnchor:row.trailingAnchor],
        [bubble.leadingAnchor constraintGreaterThanOrEqualToAnchor:row.leadingAnchor constant:52.0],

        [inner.topAnchor constraintEqualToAnchor:bubble.topAnchor constant:11.0],
        [inner.bottomAnchor constraintEqualToAnchor:bubble.bottomAnchor constant:-11.0],
        [inner.leadingAnchor constraintEqualToAnchor:bubble.leadingAnchor constant:14.0],
        [inner.trailingAnchor constraintEqualToAnchor:bubble.trailingAnchor constant:-14.0],
    ]];
    [self.messagesStack addArrangedSubview:row];
}

- (UIView *)appendKronosBubbleOutBody:(UILabel **)outBody outExtraStack:(UIStackView **)outExtra {
    UIView *card = [[UIView alloc] init];
    card.translatesAutoresizingMaskIntoConstraints = NO;

    UIStackView *vStack = [[UIStackView alloc] init];
    vStack.translatesAutoresizingMaskIntoConstraints = NO;
    vStack.axis = UILayoutConstraintAxisVertical;
    vStack.spacing = 8.0;
    [card addSubview:vStack];

    UIView *hdr = [[UIView alloc] init];
    hdr.translatesAutoresizingMaskIntoConstraints = NO;

    UIView *dot = [[UIView alloc] init];
    dot.translatesAutoresizingMaskIntoConstraints = NO;
    dot.backgroundColor = kronosAccentColor();
    dot.layer.cornerRadius = 4.0;
    [hdr addSubview:dot];

    UILabel *nameLbl = [[UILabel alloc] init];
    nameLbl.translatesAutoresizingMaskIntoConstraints = NO;
    nameLbl.text = @"Kronos";
    nameLbl.textColor = kronosTextColor();
    nameLbl.font = [UIFont systemFontOfSize:13.5 weight:UIFontWeightSemibold];
    [hdr addSubview:nameLbl];

    UIButton *speakBtn = [UIButton buttonWithType:UIButtonTypeSystem];
    speakBtn.translatesAutoresizingMaskIntoConstraints = NO;
    [speakBtn setImage:[UIImage systemImageNamed:@"speaker.wave.2"] forState:UIControlStateNormal];
    speakBtn.tintColor = kronosMutedColor();
    [speakBtn addTarget:self action:@selector(speakBubbleTapped:) forControlEvents:UIControlEventTouchUpInside];
    [hdr addSubview:speakBtn];

    UIButton *copyBtn = [UIButton buttonWithType:UIButtonTypeSystem];
    copyBtn.translatesAutoresizingMaskIntoConstraints = NO;
    [copyBtn setImage:[UIImage systemImageNamed:@"doc.on.doc"] forState:UIControlStateNormal];
    copyBtn.tintColor = kronosMutedColor();
    [copyBtn addTarget:self action:@selector(copyBubbleTapped:) forControlEvents:UIControlEventTouchUpInside];
    [hdr addSubview:copyBtn];

    [NSLayoutConstraint activateConstraints:@[
        [hdr.heightAnchor constraintEqualToConstant:24.0],
        [dot.leadingAnchor constraintEqualToAnchor:hdr.leadingAnchor],
        [dot.centerYAnchor constraintEqualToAnchor:hdr.centerYAnchor],
        [dot.widthAnchor constraintEqualToConstant:8.0],
        [dot.heightAnchor constraintEqualToConstant:8.0],

        [nameLbl.leadingAnchor constraintEqualToAnchor:dot.trailingAnchor constant:7.0],
        [nameLbl.centerYAnchor constraintEqualToAnchor:hdr.centerYAnchor],

        [copyBtn.trailingAnchor constraintEqualToAnchor:hdr.trailingAnchor],
        [copyBtn.centerYAnchor constraintEqualToAnchor:hdr.centerYAnchor],
        [copyBtn.widthAnchor constraintEqualToConstant:28.0],
        [copyBtn.heightAnchor constraintEqualToConstant:28.0],

        [speakBtn.trailingAnchor constraintEqualToAnchor:copyBtn.leadingAnchor constant:-6.0],
        [speakBtn.centerYAnchor constraintEqualToAnchor:hdr.centerYAnchor],
        [speakBtn.widthAnchor constraintEqualToConstant:28.0],
        [speakBtn.heightAnchor constraintEqualToConstant:28.0],
    ]];
    [vStack addArrangedSubview:hdr];

    UILabel *bodyLbl = [[UILabel alloc] init];
    bodyLbl.text = @"Thinking...";
    bodyLbl.textColor = kronosMutedColor();
    bodyLbl.font = [UIFont systemFontOfSize:15.5 weight:UIFontWeightRegular];
    bodyLbl.numberOfLines = 0;
    bodyLbl.tag = 3301;
    [vStack addArrangedSubview:bodyLbl];

    UIStackView *extraStack = [[UIStackView alloc] init];
    extraStack.axis = UILayoutConstraintAxisVertical;
    extraStack.spacing = 8.0;
    [vStack addArrangedSubview:extraStack];

    [NSLayoutConstraint activateConstraints:@[
        [vStack.topAnchor constraintEqualToAnchor:card.topAnchor constant:4.0],
        [vStack.bottomAnchor constraintEqualToAnchor:card.bottomAnchor constant:-4.0],
        [vStack.leadingAnchor constraintEqualToAnchor:card.leadingAnchor constant:2.0],
        [vStack.trailingAnchor constraintEqualToAnchor:card.trailingAnchor constant:-2.0],
    ]];

    [self.messagesStack addArrangedSubview:card];
    if (outBody != NULL) *outBody = bodyLbl;
    if (outExtra != NULL) *outExtra = extraStack;
    return card;
}

- (void)addActionPillWithTitle:(NSString *)title previewFile:(NSString *)previewFile toStack:(UIStackView *)stack {
    if (title.length == 0 || stack == nil) return;
    UIButton *pill = [UIButton buttonWithType:UIButtonTypeSystem];
    pill.translatesAutoresizingMaskIntoConstraints = NO;
    NSString *fullLabel = previewFile.length > 0
        ? [NSString stringWithFormat:@"  %@ · Tap to open", title]
        : [NSString stringWithFormat:@"  %@", title];
    [pill setImage:[UIImage systemImageNamed:previewFile.length > 0 ? @"doc.richtext" : @"checkmark.circle.fill"] forState:UIControlStateNormal];
    [pill setTitle:fullLabel forState:UIControlStateNormal];
    pill.tintColor = kronosAccentColor();
    [pill setTitleColor:kronosAccentColor() forState:UIControlStateNormal];
    pill.titleLabel.font = [UIFont systemFontOfSize:13.0 weight:UIFontWeightSemibold];
    pill.contentHorizontalAlignment = UIControlContentHorizontalAlignmentLeading;
    pill.backgroundColor = [kronosAccentColor() colorWithAlphaComponent:0.12];
    pill.layer.cornerRadius = 10.0;
    [pill.heightAnchor constraintEqualToConstant:36.0].active = YES;
    if (previewFile.length > 0) {
        pill.accessibilityIdentifier = previewFile;
        [pill addTarget:self action:@selector(previewPillTapped:) forControlEvents:UIControlEventTouchUpInside];
    }
    [stack addArrangedSubview:pill];
}

- (void)addPhotoPreviewCardWithImage:(UIImage *)image caption:(NSString *)caption toStack:(UIStackView *)stack {
    if (image == nil || stack == nil) return;
    UIView *card = [[UIView alloc] init];
    card.translatesAutoresizingMaskIntoConstraints = NO;
    card.backgroundColor = kronosPanelColor();
    card.layer.cornerRadius = 14.0;
    card.layer.borderWidth = 1.0;
    card.layer.borderColor = kronosLineColor().CGColor;
    card.clipsToBounds = YES;

    UIImageView *iv = [[UIImageView alloc] initWithImage:image];
    iv.translatesAutoresizingMaskIntoConstraints = NO;
    iv.contentMode = UIViewContentModeScaleAspectFill;
    iv.clipsToBounds = YES;
    [card addSubview:iv];

    UILabel *cap = [[UILabel alloc] init];
    cap.translatesAutoresizingMaskIntoConstraints = NO;
    cap.text = caption ?: @"Photo preview";
    cap.textColor = kronosMutedColor();
    cap.font = [UIFont systemFontOfSize:12.0 weight:UIFontWeightMedium];
    [card addSubview:cap];

    [NSLayoutConstraint activateConstraints:@[
        [iv.topAnchor constraintEqualToAnchor:card.topAnchor],
        [iv.leadingAnchor constraintEqualToAnchor:card.leadingAnchor],
        [iv.trailingAnchor constraintEqualToAnchor:card.trailingAnchor],
        [iv.heightAnchor constraintEqualToConstant:180.0],

        [cap.topAnchor constraintEqualToAnchor:iv.bottomAnchor constant:8.0],
        [cap.leadingAnchor constraintEqualToAnchor:card.leadingAnchor constant:12.0],
        [cap.trailingAnchor constraintEqualToAnchor:card.trailingAnchor constant:-12.0],
        [cap.bottomAnchor constraintEqualToAnchor:card.bottomAnchor constant:-8.0],
    ]];
    [stack addArrangedSubview:card];
}

- (void)addWhatsAppOpenDraftButtonToStack:(UIStackView *)stack {
    if (stack == nil) return;
    UIButton *btn = [UIButton buttonWithType:UIButtonTypeSystem];
    btn.translatesAutoresizingMaskIntoConstraints = NO;
    [btn setImage:[UIImage systemImageNamed:@"paperplane.fill"] forState:UIControlStateNormal];
    [btn setTitle:@"  Open Draft in WhatsApp (Review & Tap Send)" forState:UIControlStateNormal];
    btn.tintColor = [UIColor whiteColor];
    [btn setTitleColor:[UIColor whiteColor] forState:UIControlStateNormal];
    btn.titleLabel.font = [UIFont systemFontOfSize:13.5 weight:UIFontWeightSemibold];
    btn.backgroundColor = kronosAccentColor();
    btn.layer.cornerRadius = 12.0;
    [btn.heightAnchor constraintEqualToConstant:42.0].active = YES;
    [btn addTarget:self action:@selector(reopenWhatsAppDraftTapped) forControlEvents:UIControlEventTouchUpInside];
    [stack addArrangedSubview:btn];
}

- (void)addLinkedInPublishButtonsToStack:(UIStackView *)stack {
    if (stack == nil) return;
    UIButton *shareBtn = [UIButton buttonWithType:UIButtonTypeSystem];
    shareBtn.translatesAutoresizingMaskIntoConstraints = NO;
    [shareBtn setImage:[UIImage systemImageNamed:@"square.and.arrow.up"] forState:UIControlStateNormal];
    [shareBtn setTitle:@"  Publish on LinkedIn (Review & Post)" forState:UIControlStateNormal];
    shareBtn.tintColor = [UIColor whiteColor];
    [shareBtn setTitleColor:[UIColor whiteColor] forState:UIControlStateNormal];
    shareBtn.titleLabel.font = [UIFont systemFontOfSize:14.0 weight:UIFontWeightSemibold];
    shareBtn.backgroundColor = kronosAccentColor();
    shareBtn.layer.cornerRadius = 12.0;
    [shareBtn.heightAnchor constraintEqualToConstant:44.0].active = YES;
    [shareBtn addTarget:self action:@selector(presentLinkedInComposerForCurrentDraft) forControlEvents:UIControlEventTouchUpInside];
    [stack addArrangedSubview:shareBtn];

    UIButton *feedBtn = [UIButton buttonWithType:UIButtonTypeSystem];
    feedBtn.translatesAutoresizingMaskIntoConstraints = NO;
    [feedBtn setImage:[UIImage systemImageNamed:@"arrow.up.right.square"] forState:UIControlStateNormal];
    [feedBtn setTitle:@"  Open LinkedIn Feed (Draft Copied to Clipboard)" forState:UIControlStateNormal];
    feedBtn.tintColor = kronosAccentColor();
    [feedBtn setTitleColor:kronosAccentColor() forState:UIControlStateNormal];
    feedBtn.titleLabel.font = [UIFont systemFontOfSize:13.0 weight:UIFontWeightSemibold];
    feedBtn.backgroundColor = [kronosAccentColor() colorWithAlphaComponent:0.12];
    feedBtn.layer.cornerRadius = 11.0;
    [feedBtn.heightAnchor constraintEqualToConstant:38.0].active = YES;
    [feedBtn addTarget:self action:@selector(openLinkedInFeedTapped) forControlEvents:UIControlEventTouchUpInside];
    [stack addArrangedSubview:feedBtn];
}

- (void)reopenWhatsAppDraftTapped {
    if (self.lastWhatsAppDraftURL != nil) {
        [[UIApplication sharedApplication] openURL:self.lastWhatsAppDraftURL options:@{} completionHandler:nil];
    }
}

- (void)presentLinkedInComposerForCurrentDraft {
    NSMutableArray *items = [NSMutableArray array];
    if (self.lastLinkedInDraftText.length > 0) {
        [UIPasteboard generalPasteboard].string = self.lastLinkedInDraftText;
        [items addObject:self.lastLinkedInDraftText];
    }
    if (self.lastLinkedInDraftPhoto != nil) {
        [items addObject:self.lastLinkedInDraftPhoto];
    }
    if (items.count == 0) return;
    UIActivityViewController *avc = [[UIActivityViewController alloc] initWithActivityItems:items applicationActivities:nil];
    [self presentViewController:avc animated:YES completion:nil];
}

- (void)openLinkedInFeedTapped {
    if (self.lastLinkedInDraftText.length > 0) {
        [UIPasteboard generalPasteboard].string = self.lastLinkedInDraftText;
    }
    NSURL *appURL = [NSURL URLWithString:@"linkedin://"];
    NSURL *webURL = [NSURL URLWithString:@"https://www.linkedin.com/feed/"];
    [[UIApplication sharedApplication] openURL:appURL options:@{} completionHandler:^(BOOL ok) {
        if (!ok && webURL != nil) {
            [[UIApplication sharedApplication] openURL:webURL options:@{} completionHandler:nil];
        }
    }];
}

- (void)previewPillTapped:(UIButton *)sender {
    if (sender.accessibilityIdentifier.length > 0) {
        [self openFilePreviewForFilename:sender.accessibilityIdentifier];
    }
}

- (void)copyBubbleTapped:(UIButton *)sender {
    UIView *cur = sender;
    while (cur != nil && [cur viewWithTag:3301] == nil) cur = cur.superview;
    UILabel *body = (UILabel *)[cur viewWithTag:3301];
    if (body.text.length > 0) {
        [UIPasteboard generalPasteboard].string = body.text;
    }
}

- (void)speakBubbleTapped:(UIButton *)sender {
    UIView *cur = sender;
    while (cur != nil && [cur viewWithTag:3301] == nil) cur = cur.superview;
    UILabel *body = (UILabel *)[cur viewWithTag:3301];
    if (body.text.length > 0) {
        [self.voiceController speakText:body.text];
    }
}

#pragma mark - Submission, Agentic WhatsApp & LinkedIn Workflows, Strict Messaging Lookup & Chat Execution

- (void)recordImmediateKronosReply:(NSString *)replyText
                       actionBadge:(NSString *)badge
                         replyBody:(UILabel *)replyBody
                        replyExtra:(UIStackView *)replyExtra
                          convMsgs:(NSMutableArray *)convMsgs {
    replyBody.attributedText = kronosFormatMessageText(replyText, NO);
    if (badge.length > 0) {
        [self addActionPillWithTitle:badge previewFile:nil toStack:replyExtra];
    }
    [convMsgs addObject:@{
        @"role": @"assistant",
        @"content": replyText,
        @"displayText": replyText,
        @"actionBadge": badge ?: @"",
    }];
    self.activeConversation[@"updatedAt"] = @([[NSDate date] timeIntervalSince1970]);
    [KronosConversationStore saveConversations:self.conversations];
    if (self.voiceController.voiceOutputEnabled || self.lastTurnWasSpoken) {
        [self.voiceController speakText:replyText];
    }
    [self scrollToBottomAnimated:YES];
}

- (void)completeWhatsAppDraftForRecipient:(NSString *)recipient
                              rawUserSpec:(NSString *)rawUserSpec
                            isInstruction:(BOOL)isInstruction
                                replyBody:(UILabel *)replyBody
                               replyExtra:(UIStackView *)replyExtra
                                 convMsgs:(NSMutableArray *)convMsgs {
    if (!isInstruction) {
        NSString *directMsg = kronosCleanDraftOutput(rawUserSpec);
        if (directMsg.length > 0) {
            NSString *first = [[directMsg substringToIndex:1] uppercaseString];
            directMsg = [first stringByAppendingString:[directMsg substringFromIndex:1]];
        }
        [self launchWhatsAppWithPreparedDraft:directMsg
                                    recipient:recipient
                                    replyBody:replyBody
                                   replyExtra:replyExtra
                                     convMsgs:convMsgs];
        return;
    }

    self.isGenerating = YES;
    self.headerSubtitleLabel.text = @"Drafting WhatsApp message...";
    [self updateComposerControls];

    dispatch_async(self.engineQueue, ^{
        NSString *composed = nil;
        if (self.session != NULL) {
            NSString *sys = @"You draft concise, natural WhatsApp messages. Output ONLY the ready-to-send message body itself in 1 to 2 sentences. Do not include quotation marks, explanations, or sign-off placeholders.";
            NSString *usr = [NSString stringWithFormat:@"Write a WhatsApp message to %@ for this request: %@", recipient, rawUserSpec];
            NSArray *msgs = @[@{@"role": @"system", @"content": sys}, @{@"role": @"user", @"content": usr}];
            NSData *jd = [NSJSONSerialization dataWithJSONObject:msgs options:0 error:nil];
            NSString *jStr = [[NSString alloc] initWithData:jd encoding:NSUTF8StringEncoding];
            NSMutableString *acc = [NSMutableString string];
            KronosStreamCtx sctx = { .buffer = acc, .onFlush = nil, .cancelFlag = &_cancelFlag, .lastFlushMs = kronos_monotonic_ms() };
            double pf = 0.0, dc = 0.0;
            size_t pft = 0;
            kronos_session_generate_chat_json(self.session, [jStr UTF8String], 120, 0.2f, kronos_chat_stream_cb, &sctx, &pf, &dc, &pft);
            composed = kronosCleanDraftOutput(acc);
        }
        if (composed.length < 4) {
            composed = [NSString stringWithFormat:@"Hi %@, hope you are doing well. Wanted to check in regarding %@.", recipient, rawUserSpec];
        }
        dispatch_async(dispatch_get_main_queue(), ^{
            self.isGenerating = NO;
            self.headerSubtitleLabel.text = @"";
            [self updateComposerControls];
            [self launchWhatsAppWithPreparedDraft:composed
                                        recipient:recipient
                                        replyBody:replyBody
                                       replyExtra:replyExtra
                                         convMsgs:convMsgs];
        });
    });
}

- (void)launchWhatsAppWithPreparedDraft:(NSString *)draftMessage
                              recipient:(NSString *)recipient
                              replyBody:(UILabel *)replyBody
                             replyExtra:(UIStackView *)replyExtra
                               convMsgs:(NSMutableArray *)convMsgs {
    self.headerSubtitleLabel.text = @"Opening WhatsApp draft...";
    kronosOpenWhatsAppDraftForRecipient(recipient, draftMessage, ^(BOOL opened, NSString *displayRecipient, BOOL matchedContactPhone, NSURL *draftURL) {
        self.headerSubtitleLabel.text = @"";
        self.lastWhatsAppDraftURL = draftURL;

        NSString *reply = nil;
        if (opened && matchedContactPhone) {
            reply = [NSString stringWithFormat:
                     @"I prepared your WhatsApp message for %@ and opened WhatsApp with the draft pre-filled (not sent):\n\n\"%@\"\n\nPlease review the message in WhatsApp and tap Send to submit it.",
                     displayRecipient, draftMessage];
        } else if (opened) {
            reply = [NSString stringWithFormat:
                     @"I prepared your WhatsApp message for %@ (and copied it to your clipboard) and opened WhatsApp with the draft pre-filled, not sent:\n\n\"%@\"\n\nSelect %@ in WhatsApp, review the message in the input box, and tap Send to submit it.",
                     displayRecipient, draftMessage, displayRecipient];
        } else {
            reply = [NSString stringWithFormat:
                     @"I drafted your WhatsApp message for %@ and copied it to your clipboard (not sent):\n\n\"%@\"\n\nOpen WhatsApp, paste or review the draft for %@, and tap Send to submit it.",
                     displayRecipient, draftMessage, displayRecipient];
        }

        replyBody.attributedText = kronosFormatMessageText(reply, NO);
        [self addWhatsAppOpenDraftButtonToStack:replyExtra];
        [convMsgs addObject:@{
            @"role": @"assistant",
            @"content": reply,
            @"displayText": reply,
            @"actionType": @"whatsapp",
            @"draftURL": draftURL.absoluteString ?: @"",
        }];
        self.activeConversation[@"updatedAt"] = @([[NSDate date] timeIntervalSince1970]);
        [KronosConversationStore saveConversations:self.conversations];
        if (self.voiceController.voiceOutputEnabled || self.lastTurnWasSpoken) {
            [self.voiceController speakText:[NSString stringWithFormat:@"I prepared your WhatsApp draft for %@. Please review it in WhatsApp and tap Send to submit it.", displayRecipient]];
        }
        [self scrollToBottomAnimated:YES];
    });
}

- (void)handleLinkedInPostWorkflowWithTopic:(NSString *)topic
                                 wantsPhoto:(BOOL)wantsPhoto
                             attachedThumbs:(NSArray<UIImage *> *)attachedThumbs
                                  replyBody:(UILabel *)replyBody
                                 replyExtra:(UIStackView *)replyExtra
                                   convMsgs:(NSMutableArray *)convMsgs {
    self.isGenerating = YES;
    self.headerSubtitleLabel.text = wantsPhoto ? @"Fetching today's latest photo..." : @"Drafting LinkedIn post...";
    [self updateComposerControls];

    void (^generateWithPhoto)(UIImage *, NSDate *, BOOL, NSString *) = ^(UIImage *photo, NSDate *takenDate, BOOL takenToday, NSString *photoNote) {
        self.headerSubtitleLabel.text = @"Drafting LinkedIn post...";
        UIImage *uprightPhoto = (photo != nil) ? kronosRenderUprightImage(photo, 1280.0) : nil;
        NSString *savedPhotoFilename = nil;
        if (uprightPhoto != nil) {
            NSData *jpg = UIImageJPEGRepresentation(uprightPhoto, 0.88);
            if (jpg.length > 0) {
                savedPhotoFilename = [NSString stringWithFormat:@"linkedin_photo_%ld.jpg", (long)[[NSDate date] timeIntervalSince1970]];
                NSString *dest = [kronosWorkspaceDirectory() stringByAppendingPathComponent:savedPhotoFilename];
                [jpg writeToFile:dest atomically:YES];
                [self refreshWorkspaceFileCount];
            }
        }

        dispatch_async(self.engineQueue, ^{
            NSString *photoVisualSummary = (uprightPhoto != nil) ? kronosSummarizePhotoSubjects(uprightPhoto) : nil;
            NSString *draftPost = nil;
            if (self.session != NULL) {
                NSString *sys = @"You write authentic, thoughtful first-person LinkedIn posts for Melody Leonard, a Senior Software Engineer with over 10 years of software engineering experience. Write ONLY the LinkedIn post text itself in 2 short paragraphs followed by 3 relevant hashtags. Do not use markdown bold asterisks and do not add introductory commentary.";
                NSMutableString *usr = [NSMutableString stringWithFormat:@"Write a LinkedIn post about %@.", topic];
                if (photoVisualSummary.length > 0) {
                    [usr appendFormat:@" The attached photo taken today shows: %@.", photoVisualSummary];
                }
                NSArray *msgs = @[@{@"role": @"system", @"content": sys}, @{@"role": @"user", @"content": usr}];
                NSData *jd = [NSJSONSerialization dataWithJSONObject:msgs options:0 error:nil];
                NSString *jStr = [[NSString alloc] initWithData:jd encoding:NSUTF8StringEncoding];
                NSMutableString *acc = [NSMutableString string];
                KronosStreamCtx sctx = { .buffer = acc, .onFlush = nil, .cancelFlag = &_cancelFlag, .lastFlushMs = kronos_monotonic_ms() };
                double pf = 0.0, dc = 0.0;
                size_t pft = 0;
                kronos_session_generate_chat_json(self.session, [jStr UTF8String], 240, 0.25f, kronos_chat_stream_cb, &sctx, &pf, &dc, &pft);
                draftPost = kronosCleanDraftOutput(acc);
            }
            if (draftPost.length < 45) {
                draftPost = [NSString stringWithFormat:
                             @"Starting the week focused on %@ and steady engineering execution.\n\nConsistency in shipping clean, reliable systems one iteration at a time always compounds faster than waiting for perfect conditions. Wishing everyone building this week a productive start.\n\n#MondayMotivation #SoftwareEngineering #SystemsArchitecture",
                             topic];
            }

            dispatch_async(dispatch_get_main_queue(), ^{
                self.isGenerating = NO;
                self.headerSubtitleLabel.text = @"";
                [self updateComposerControls];

                self.lastLinkedInDraftText = draftPost;
                self.lastLinkedInDraftPhoto = uprightPhoto;
                [UIPasteboard generalPasteboard].string = draftPost;

                NSMutableString *fullReply = [NSMutableString string];
                if (uprightPhoto != nil) {
                    NSDateFormatter *df = [[NSDateFormatter alloc] init];
                    df.dateFormat = takenToday ? @"h:mm a" : @"MMM d, h:mm a";
                    NSString *timeStr = [df stringFromDate:(takenDate ?: [NSDate date])];
                    if (takenToday) {
                        [fullReply appendFormat:@"I selected the last picture you took today (%@) and drafted your LinkedIn post:\n\n", timeStr];
                    } else {
                        [fullReply appendFormat:@"No photo taken earlier today was found, so I attached your most recent photo (%@) and drafted your LinkedIn post:\n\n", timeStr];
                    }
                } else if (photoNote.length > 0) {
                    [fullReply appendFormat:@"%@\n\nHere is your LinkedIn post draft:\n\n", photoNote];
                } else {
                    [fullReply appendString:@"Here is your LinkedIn post draft:\n\n"];
                }
                [fullReply appendFormat:@"%@\n\n", draftPost];
                [fullReply appendString:@"I have not published this post automatically. Your draft is also copied to your clipboard — please review it and choose LinkedIn in the share sheet (or tap Publish on LinkedIn below) when you are ready to publish it."];

                replyBody.attributedText = kronosFormatMessageText(fullReply, NO);
                if (uprightPhoto != nil) {
                    NSString *cap = takenToday ? @"Last picture taken today · Ready for LinkedIn" : @"Most recent photo in library · Ready for LinkedIn";
                    [self addPhotoPreviewCardWithImage:uprightPhoto caption:cap toStack:replyExtra];
                }
                [self addLinkedInPublishButtonsToStack:replyExtra];

                [convMsgs addObject:@{
                    @"role": @"assistant",
                    @"content": fullReply,
                    @"displayText": fullReply,
                    @"actionType": @"linkedin",
                    @"draftPostText": draftPost,
                    @"previewImageFile": savedPhotoFilename ?: @"",
                }];
                self.activeConversation[@"updatedAt"] = @([[NSDate date] timeIntervalSince1970]);
                [KronosConversationStore saveConversations:self.conversations];

                if (self.voiceController.voiceOutputEnabled || self.lastTurnWasSpoken) {
                    [self.voiceController speakText:@"I created your LinkedIn post with your latest photo. Please review the draft and tap Publish on LinkedIn when you are ready to post it."];
                }
                [self scrollToBottomAnimated:YES];
                [self presentLinkedInComposerForCurrentDraft];
            });
        });
    };

    if (attachedThumbs.count > 0) {
        generateWithPhoto(attachedThumbs.firstObject, [NSDate date], YES, nil);
    } else if (wantsPhoto) {
        [[KronosIosDeviceContextManager sharedManager] fetchLatestPhotoForPostWithCompletion:^(UIImage *image, NSDate *takenDate, BOOL takenToday, NSString *errorNote) {
            generateWithPhoto(image, takenDate, takenToday, errorNote);
        }];
    } else {
        generateWithPhoto(nil, nil, NO, nil);
    }
}

- (void)submitUserPrompt:(NSString *)userText {
    if (self.welcomeView != nil) {
        [self.messagesStack removeArrangedSubview:self.welcomeView];
        [self.welcomeView removeFromSuperview];
        self.welcomeView = nil;
    }

    NSArray<KronosAttachmentItem *> *turnAttachments = [self.pendingAttachments copy];
    [self.pendingAttachments removeAllObjects];
    [self rebuildPendingAttachmentsStrip];

    NSMutableArray<NSString *> *attNames = [NSMutableArray array];
    NSMutableArray<UIImage *> *thumbs = [NSMutableArray array];
    NSMutableString *attPromptBlock = [NSMutableString string];
    NSMutableString *fullModelPrompt = [NSMutableString string];

    for (KronosAttachmentItem *att in turnAttachments) {
        [attNames addObject:att.filename];
        if (att.isImage && att.thumbnail != nil) [thumbs addObject:att.thumbnail];
        if (att.promptContext.length > 0) {
            [attPromptBlock appendString:att.promptContext];
            [attPromptBlock appendString:@"\n"];
        }
    }
    if (attPromptBlock.length > 0) {
        [fullModelPrompt appendString:attPromptBlock];
    }

    NSMutableArray *convMsgs = self.activeConversation[@"messages"];
    BOOL awaitingWhatsAppDraftAnswer = (turnAttachments.count == 0) &&
        (self.pendingWhatsAppRecipient.length > 0 || self.pendingWhatsAppAwaitingRecipient);
    NSString *effectiveQuery = (!awaitingWhatsAppDraftAnswer && turnAttachments.count == 0)
        ? kronosResolveFollowUpUserQuery(userText, convMsgs)
        : userText;

    BOOL asksProfile = (turnAttachments.count == 0) && kronosQueryAsksCreatorOrProfile(effectiveQuery);
    BOOL asksMemory = (turnAttachments.count == 0) && kronosQueryAsksConversationMemory(effectiveQuery);
    BOOL hasLearnedMemBlock = NO;
    if (turnAttachments.count == 0) {
        NSString *learnedMem = kronosBuildLearnedMemoryPromptBlock(effectiveQuery);
        if (learnedMem.length > 0) {
            [fullModelPrompt appendString:learnedMem];
            [fullModelPrompt appendString:@"\n"];
            hasLearnedMemBlock = YES;
        }
    }
    if (asksMemory) {
        NSString *memCtx = [KronosConversationStore buildMemoryContextForQuery:effectiveQuery
                                                                 conversations:self.conversations
                                                                      activeId:self.activeConversation[@"id"]];
        [fullModelPrompt appendString:memCtx];
        [fullModelPrompt appendString:@"\n"];
    }
    if (turnAttachments.count == 0) {
        NSString *devCtx = [[KronosIosDeviceContextManager sharedManager] buildDeviceContextPromptBlockIfRelevant:effectiveQuery];
        if (devCtx.length > 0) {
            [fullModelPrompt appendString:devCtx];
            [fullModelPrompt appendString:@"\n"];
        }
    }
    if (asksProfile) {
        [fullModelPrompt appendString:kKronosCreatorProfileFacts];
        [fullModelPrompt appendString:@"\n\nUser question: "];
    } else if (asksMemory || hasLearnedMemBlock || turnAttachments.count > 0) {
        [fullModelPrompt appendString:@"User question: "];
    }
    [fullModelPrompt appendString:effectiveQuery];

    [self appendUserBubbleWithText:userText attachmentNames:attNames thumbnails:thumbs];
    UILabel *replyBody = nil;
    UIStackView *replyExtra = nil;
    [self appendKronosBubbleOutBody:&replyBody outExtraStack:&replyExtra];
    [self scrollToBottomAnimated:YES];

    if (convMsgs.count == 0) {
        NSString *shortTitle = userText.length > 34 ? [[userText substringToIndex:34] stringByAppendingString:@"..."] : userText;
        self.activeConversation[@"title"] = shortTitle;
        self.headerTitleLabel.text = shortTitle;
    }
    [convMsgs addObject:@{
        @"role": @"user",
        @"displayText": userText,
        @"content": userText,
        @"resolvedQuery": effectiveQuery ?: userText,
        @"attachmentContext": [attPromptBlock copy],
        @"attachments": attNames,
    }];
    self.activeConversation[@"updatedAt"] = @([[NSDate date] timeIntervalSince1970]);
    [KronosConversationStore saveConversations:self.conversations];
    [self applyConversationSearchFilter];

    NSString *trimmedLower = [[kronosNormalizeApostrophes(userText) stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]] lowercaseString];
    NSString *effectiveLower = [[kronosNormalizeApostrophes(effectiveQuery) stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]] lowercaseString];

    // 1. Handle follow-up clarification turn for a pending WhatsApp draft.
    if (turnAttachments.count == 0 && (self.pendingWhatsAppRecipient.length > 0 || self.pendingWhatsAppAwaitingRecipient)) {
        BOOL isCancel = [trimmedLower isEqualToString:@"cancel"] || [trimmedLower isEqualToString:@"never mind"] ||
                        [trimmedLower isEqualToString:@"nevermind"] || [trimmedLower isEqualToString:@"stop"];
        if (isCancel) {
            NSString *cancelledFor = self.pendingWhatsAppRecipient ?: @"WhatsApp";
            self.pendingWhatsAppRecipient = nil;
            self.pendingWhatsAppAwaitingRecipient = NO;
            [self recordImmediateKronosReply:[NSString stringWithFormat:@"Cancelled the WhatsApp draft for %@.", cancelledFor]
                                 actionBadge:@"Draft Cancelled"
                                   replyBody:replyBody
                                  replyExtra:replyExtra
                                    convMsgs:convMsgs];
            return;
        }

        BOOL startsNewCommandOrQuestion = kronosIsInterrogativeOrLookupText(userText) ||
                                          kronosParseLinkedInPostIntent(userText, NULL, NULL) ||
                                          kronosParseWhatsAppDraftIntent(userText, NULL, NULL, NULL, NULL, NULL) ||
                                          [trimmedLower hasPrefix:@"open github"] || [trimmedLower hasPrefix:@"open safari"];
        if (startsNewCommandOrQuestion) {
            self.pendingWhatsAppRecipient = nil;
            self.pendingWhatsAppAwaitingRecipient = NO;
            effectiveQuery = kronosResolveFollowUpUserQuery(userText, [convMsgs subarrayWithRange:NSMakeRange(0, convMsgs.count > 0 ? convMsgs.count - 1 : 0)]);
            effectiveLower = [[kronosNormalizeApostrophes(effectiveQuery) stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]] lowercaseString];
        } else if (self.pendingWhatsAppAwaitingRecipient) {
            self.pendingWhatsAppAwaitingRecipient = NO;
            NSString *cleanRecip = kronosCleanExtractedSpan(userText);
            self.pendingWhatsAppRecipient = cleanRecip;
            NSString *askMsg = [NSString stringWithFormat:@"What kind of message would you like me to send to %@ on WhatsApp?", cleanRecip];
            [self recordImmediateKronosReply:askMsg
                                 actionBadge:[NSString stringWithFormat:@"Awaiting message details for %@", cleanRecip]
                                   replyBody:replyBody
                                  replyExtra:replyExtra
                                    convMsgs:convMsgs];
            return;
        } else {
            NSString *recipient = self.pendingWhatsAppRecipient;
            self.pendingWhatsAppRecipient = nil;
            BOOL isInstr = kronosIsWhatsAppDraftInstructionStyle(userText);
            [self completeWhatsAppDraftForRecipient:recipient
                                        rawUserSpec:userText
                                      isInstruction:isInstr
                                          replyBody:replyBody
                                         replyExtra:replyExtra
                                           convMsgs:convMsgs];
            return;
        }
    }

    // 1b. Save user-taught local personal facts, preferences, corrections, and reminders
    //     ("This is my WhatsApp number 09033351286", "My birthday is ...", "Remind me to ...", "Remember ...").
    NSString *savedFactConfirm = nil;
    if (turnAttachments.count == 0 && kronosTrySaveUserTaughtFact(userText, &savedFactConfirm)) {
        kronosSyncContinualMemoryAndDistillVaultWithMac();
        [self recordImmediateKronosReply:savedFactConfirm
                             actionBadge:@"Saved to Local Memory"
                               replyBody:replyBody
                              replyExtra:replyExtra
                                convMsgs:convMsgs];
        return;
    }

    // 1b-recall. Direct recall of user-taught facts and reminders ("What is my WhatsApp number?", "Remind me of my WhatsApp number", "What are my reminders?").
    BOOL looksLikeFactRecall = [effectiveLower hasSuffix:@"?"] ||
                               [effectiveLower hasPrefix:@"what "] || [effectiveLower hasPrefix:@"what's "] ||
                               [effectiveLower hasPrefix:@"whats "] || [effectiveLower hasPrefix:@"when "] ||
                               [effectiveLower hasPrefix:@"where "] || [effectiveLower hasPrefix:@"remind me "] ||
                               [effectiveLower hasPrefix:@"do you remember "] || [effectiveLower hasPrefix:@"do you know "] ||
                               [effectiveLower hasPrefix:@"tell me my "] || [effectiveLower hasPrefix:@"show me my "];
    if (turnAttachments.count == 0 && !asksProfile && looksLikeFactRecall &&
        !kronosIsActionOrComposerCommand(effectiveLower) && !kronosIsExternalMessageLookupQuery(effectiveQuery)) {
        NSString *directLearnedFact = kronosLookupVerifiedLocalFact(effectiveQuery);
        if (directLearnedFact != nil) {
            [self recordImmediateKronosReply:directLearnedFact
                                 actionBadge:@"Verified Local Memory"
                                   replyBody:replyBody
                                  replyExtra:replyExtra
                                    convMsgs:convMsgs];
            return;
        }
    }

    // 1c. Deterministic iPhone Contacts lookup ("What's my Dad's phone number", "What is Sandra's phone number").
    NSString *contactTarget = nil;
    if (turnAttachments.count == 0 && kronosIsContactPhoneOrEmailQuery(effectiveQuery, &contactTarget)) {
        NSString *taughtContactFact = kronosLookupVerifiedLocalFact(effectiveQuery);
        if (taughtContactFact == nil && contactTarget.length > 0) {
            taughtContactFact = kronosLookupVerifiedLocalFact([NSString stringWithFormat:@"%@ phone number", contactTarget]);
        }
        if (taughtContactFact != nil) {
            [self recordImmediateKronosReply:taughtContactFact
                                 actionBadge:@"Verified Local Fact"
                                   replyBody:replyBody
                                  replyExtra:replyExtra
                                    convMsgs:convMsgs];
            return;
        }
        self.headerSubtitleLabel.text = @"Checking iPhone Contacts...";
        [[KronosIosDeviceContextManager sharedManager] lookupContactsForTarget:contactTarget completion:^(BOOL found, NSString *directReply, NSString *primaryCallDigits) {
            (void)primaryCallDigits;
            self.headerSubtitleLabel.text = @"";
            [self recordImmediateKronosReply:directReply
                                 actionBadge:(found ? @"Verified iPhone Contact" : @"Checked iPhone Contacts")
                                   replyBody:replyBody
                                  replyExtra:replyExtra
                                    convMsgs:convMsgs];
        }];
        return;
    }

    // 1d. Zero-trust personal identifier evidence gate ("What is my MYN", "What is my account number").
    NSString *factTopic = nil;
    if (turnAttachments.count == 0 && kronosIsUnverifiedPersonalFactQuery(effectiveQuery, &factTopic)) {
        NSString *verifiedFact = kronosLookupVerifiedLocalFact(factTopic);
        if (verifiedFact != nil) {
            [self recordImmediateKronosReply:verifiedFact
                                 actionBadge:@"Verified Local Fact"
                                   replyBody:replyBody
                                  replyExtra:replyExtra
                                    convMsgs:convMsgs];
            return;
        }
        NSString *abstainReply = [NSString stringWithFormat:
                                  @"I checked your local iPhone context (Contacts, Workspace files, and saved Kronos memory) and do not have \"%@\" stored on your device yet, so I will not guess a fake number.\n\nYou can tell me \"Remember my %@ is ...\" or attach a file containing it to store it locally on your iPhone.",
                                  factTopic, factTopic];
        [self recordImmediateKronosReply:abstainReply
                             actionBadge:@"No Local Record Found"
                               replyBody:replyBody
                              replyExtra:replyExtra
                                convMsgs:convMsgs];
        return;
    }

    // 2. Check new WhatsApp message drafting intent ("Open WhatsApp and write a message to Dera").
    NSString *waRecipient = nil;
    NSString *waExplicitMsg = nil;
    BOOL waIsTopic = NO;
    BOOL waNeedsClarification = NO;
    BOOL waNeedsRecipient = NO;
    if (turnAttachments.count == 0 &&
        kronosParseWhatsAppDraftIntent(effectiveQuery, &waRecipient, &waExplicitMsg, &waIsTopic, &waNeedsClarification, &waNeedsRecipient)) {
        if (![KronosMobileSettings sharedSettings].allowPhoneUse) {
            [self recordImmediateKronosReply:@"Allow Phone Use is currently turned off in Kronos Settings. Tap the gear icon in the top bar or sidebar to enable Phone Use so I can prepare WhatsApp drafts."
                                 actionBadge:@"Phone Use Disabled"
                                   replyBody:replyBody
                                  replyExtra:replyExtra
                                    convMsgs:convMsgs];
            return;
        }
        if (waNeedsRecipient) {
            self.pendingWhatsAppAwaitingRecipient = YES;
            [self recordImmediateKronosReply:@"Who would you like me to message on WhatsApp?"
                                 actionBadge:@"Awaiting WhatsApp contact"
                                   replyBody:replyBody
                                  replyExtra:replyExtra
                                    convMsgs:convMsgs];
            return;
        }
        if (waNeedsClarification) {
            self.pendingWhatsAppRecipient = waRecipient;
            NSString *clarify = [NSString stringWithFormat:@"What kind of message would you like me to send to %@ on WhatsApp?", waRecipient];
            [self recordImmediateKronosReply:clarify
                                 actionBadge:[NSString stringWithFormat:@"Awaiting message details for %@", waRecipient]
                                   replyBody:replyBody
                                  replyExtra:replyExtra
                                    convMsgs:convMsgs];
            return;
        }
        [self completeWhatsAppDraftForRecipient:waRecipient
                                    rawUserSpec:waExplicitMsg
                                  isInstruction:waIsTopic
                                      replyBody:replyBody
                                     replyExtra:replyExtra
                                       convMsgs:convMsgs];
        return;
    }

    // 3. Check LinkedIn post creation intent ("Create a LinkedIn post with the last picture I took today and talk about Monday motivation").
    NSString *liTopic = nil;
    BOOL liWantsPhoto = NO;
    if (kronosParseLinkedInPostIntent(effectiveQuery, &liTopic, &liWantsPhoto)) {
        if (![KronosMobileSettings sharedSettings].allowPhoneUse) {
            [self recordImmediateKronosReply:@"Allow Phone Use is currently turned off in Kronos Settings. Tap the gear icon in the top bar or sidebar to enable Phone Use so I can prepare LinkedIn posts with your photos."
                                 actionBadge:@"Phone Use Disabled"
                                   replyBody:replyBody
                                  replyExtra:replyExtra
                                    convMsgs:convMsgs];
            return;
        }
        [self handleLinkedInPostWorkflowWithTopic:liTopic
                                       wantsPhoto:(liWantsPhoto || thumbs.count > 0)
                                   attachedThumbs:thumbs
                                        replyBody:replyBody
                                       replyExtra:replyExtra
                                         convMsgs:convMsgs];
        return;
    }

    // 4. Check immediate iOS app / URL device actions ("Open WhatsApp", "Open GitHub", "Open LinkedIn").
    if (turnAttachments.count == 0) {
        kronosTryExecuteIosDeviceAction(effectiveQuery, ^(BOOL handled, NSString *actionSummary, NSString *spokenReply) {
            if (handled) {
                [self recordImmediateKronosReply:spokenReply
                                     actionBadge:actionSummary
                                       replyBody:replyBody
                                      replyExtra:replyExtra
                                        convMsgs:convMsgs];
                return;
            }

            // 5. Check strict contact-attributed external message lookup ("What was the last message Dera sent me", "How about Oliver-ke").
            if (kronosIsExternalMessageLookupQuery(effectiveQuery)) {
                self.headerSubtitleLabel.text = @"Checking verified messages...";
                [KronosConversationStore resolveExternalMessageQuery:effectiveQuery completion:^(BOOL foundVerifiedContext, NSString *verifiedContextOrDirectReply) {
                    self.headerSubtitleLabel.text = @"";
                    if (!foundVerifiedContext) {
                        BOOL isDirectMatch = [[verifiedContextOrDirectReply lowercaseString] hasPrefix:@"latest "];
                        [self recordImmediateKronosReply:verifiedContextOrDirectReply
                                             actionBadge:(isDirectMatch ? @"Verified WhatsApp Message" : @"Strict Contact Attribution")
                                               replyBody:replyBody
                                              replyExtra:replyExtra
                                                convMsgs:convMsgs];
                        return;
                    }
                    NSString *grounded = [NSString stringWithFormat:
                                          @"%@\nAnswer the user's question strictly using ONLY the verified messages above. Never guess or attribute another contact's message.\nUser question: %@",
                                          verifiedContextOrDirectReply, effectiveQuery];
                    [self runLocalModelTurnWithUserText:effectiveQuery
                                         groundedPrompt:grounded
                                            lowTempMode:YES
                                              replyBody:replyBody
                                             replyExtra:replyExtra];
                }];
                return;
            }

            [self runLocalModelTurnWithUserText:effectiveQuery
                                 groundedPrompt:fullModelPrompt
                                    lowTempMode:asksProfile
                                      replyBody:replyBody
                                     replyExtra:replyExtra];
        });
        return;
    }

    [self runLocalModelTurnWithUserText:effectiveQuery
                         groundedPrompt:fullModelPrompt
                            lowTempMode:YES
                              replyBody:replyBody
                             replyExtra:replyExtra];
}

- (void)runLocalModelTurnWithUserText:(NSString *)userText
                       groundedPrompt:(NSString *)groundedPrompt
                          lowTempMode:(BOOL)lowTempMode
                            replyBody:(UILabel *)replyBody
                           replyExtra:(UIStackView *)replyExtra {
    NSMutableArray *convMsgs = self.activeConversation[@"messages"];
    NSString *convId = self.activeConversation[@"id"] ?: @"default";

    // Build clean, role-alternating conversational history pairs, retaining verified factual answers
    // while excluding app-launch action cards, abstention banners, and poisoned legacy turns.
    NSMutableArray<NSArray<NSString *> *> *cleanPairs = [NSMutableArray array];
    NSUInteger histEnd = (convMsgs.count > 0) ? (convMsgs.count - 1) : 0;
    NSUInteger idx = 0;
    while (idx + 1 < histEnd) {
        NSDictionary *um = convMsgs[idx];
        if (![um[@"role"] isEqualToString:@"user"]) {
            idx++;
            continue;
        }
        NSDictionary *am = convMsgs[idx + 1];
        if ([am[@"role"] isEqualToString:@"user"]) {
            idx++;
            continue;
        }
        idx += 2;

        NSString *aType = am[@"actionType"] ?: @"";
        NSString *aBadge = am[@"actionBadge"] ?: @"";
        NSString *pFile = am[@"previewFile"] ?: @"";
        BOOL isVerifiedFactBadge = [aBadge isEqualToString:@"Verified WhatsApp Message"] ||
                                   [aBadge isEqualToString:@"Verified Local Memory"] ||
                                   [aBadge isEqualToString:@"Verified Local Fact"] ||
                                   [aBadge isEqualToString:@"Verified iPhone Contact"] ||
                                   [aBadge isEqualToString:@"Saved to Local Memory"];
        BOOL isSyntheticUiCard = (aType.length > 0) || (aBadge.length > 0 && pFile.length == 0 && !isVerifiedFactBadge);
        NSString *aText = am[@"displayText"] ?: (am[@"content"] ?: @"");
        if (isSyntheticUiCard || kronosIsPoisonedHistoryReply(aText)) {
            continue;
        }

        NSString *uRaw = um[@"displayText"] ?: (um[@"content"] ?: @"");
        NSString *uResolved = [um[@"resolvedQuery"] isKindOfClass:[NSString class]] ? um[@"resolvedQuery"] : @"";
        NSString *uClean = (uResolved.length > 0) ? uResolved : uRaw;
        NSString *attCtx = um[@"attachmentContext"] ?: @"";
        if (attCtx.length == 0) {
            NSString *rawLegacy = um[@"content"] ?: @"";
            if ([rawLegacy hasPrefix:@"[Attached "]) {
                attCtx = rawLegacy.length > 2400 ? [rawLegacy substringToIndex:2400] : rawLegacy;
            }
        }
        // Preserve attachment text only on the 2 most recent historical turns so follow-up file questions work without bloating long threads.
        BOOL isRecentTurn = (idx + 4 >= histEnd);
        NSString *uForHistory = (isRecentTurn && attCtx.length > 0)
            ? [NSString stringWithFormat:@"%@\nUser question: %@", (attCtx.length > 2400 ? [attCtx substringToIndex:2400] : attCtx), uClean]
            : uClean;
        if (uForHistory.length > 0 && aText.length > 0) {
            [cleanPairs addObject:@[uForHistory, aText]];
        }
    }

    NSUInteger maxHistoryPairs = 6;
    NSUInteger windowStart = (cleanPairs.count > maxHistoryPairs) ? (cleanPairs.count - maxHistoryPairs) : 0;
    NSMutableArray<NSDictionary *> *payload = [NSMutableArray arrayWithCapacity:(maxHistoryPairs * 2 + 2)];
    [payload addObject:@{@"role": @"system", @"content": kKronosMobileSystemPrompt}];
    for (NSUInteger p = windowStart; p < cleanPairs.count; p++) {
        NSArray<NSString *> *pair = cleanPairs[p];
        [payload addObject:@{@"role": @"user", @"content": pair[0]}];
        [payload addObject:@{@"role": @"assistant", @"content": pair[1]}];
    }
    NSString *finalUserPrompt = groundedPrompt.length > 0 ? groundedPrompt : (userText ?: @"");
    [payload addObject:@{@"role": @"user", @"content": finalUserPrompt}];

    NSData *jsonData = [NSJSONSerialization dataWithJSONObject:payload options:0 error:nil];
    NSString *messagesJson = [[NSString alloc] initWithData:jsonData encoding:NSUTF8StringEncoding];

    BOOL isAgentTurn = kronosIsAgenticWorkspacePrompt(userText);
    atomic_store(&_cancelFlag, false);
    self.isGenerating = YES;
    self.headerSubtitleLabel.text = isAgentTurn ? @"Working on files..." : @"Thinking...";
    [self updateComposerControls];

    dispatch_async(self.engineQueue, ^{
        while (self.session == NULL && !atomic_load(&_cancelFlag)) {
            [NSThread sleepForTimeInterval:0.05];
        }
        if (self.session == NULL) return;

        // Reset quantized KV cache whenever switching sessions or advancing the sliding history window.
        if (![convId isEqualToString:(self.kvSessionConversationId ?: @"")] || windowStart != self.kvWindowBaseTurnIndex) {
            kronos_session_reset(self.session);
            self.kvSessionConversationId = convId;
            self.kvWindowBaseTurnIndex = windowStart;
        }

        NSString *wsDir = kronosWorkspaceDirectory();
        NSSet<NSString *> *beforeFiles = [NSSet setWithArray:([[NSFileManager defaultManager] contentsOfDirectoryAtPath:wsDir error:nil] ?: @[])];

        NSMutableString *acc = [NSMutableString string];
        void (^flushBlock)(NSString *) = ^(NSString *snap) {
            NSString *cleanSnap = kronosSanitizeFirstPersonReply(snap);
            dispatch_async(dispatch_get_main_queue(), ^{
                replyBody.attributedText = kronosFormatMessageText(cleanSnap, NO);
                [self scrollToBottomAnimated:NO];
            });
        };

        KronosStreamCtx sctx;
        sctx.buffer = acc;
        sctx.onFlush = flushBlock;
        sctx.cancelFlag = &_cancelFlag;
        sctx.lastFlushMs = kronos_monotonic_ms();

        NSString *finalText = nil;
        BOOL usedMac14B = NO;
        if (isAgentTurn) {
            char finalBuf[16384] = {0};
            int32_t rc = kronos_session_run_agent_turn(
                self.session,
                [wsDir UTF8String],
                [messagesJson UTF8String],
                4,
                768,
                0.15f,
                kronos_chat_stream_cb,
                &sctx,
                finalBuf,
                sizeof(finalBuf)
            );
            if (rc > 0 && strlen(finalBuf) > 0) {
                finalText = [NSString stringWithUTF8String:finalBuf];
            } else {
                finalText = [acc copy];
            }
        } else {
            float temp = lowTempMode ? 0.0f : 0.2f;
            NSString *macReply = kronosTryGenerateChatViaLocalMac14B(payload, 512, temp);
            if (macReply.length > 0 && !atomic_load(&_cancelFlag)) {
                finalText = macReply;
                usedMac14B = YES;
            } else {
                NSArray<NSDictionary *> *teacherShots = kronosRetrieveDistilledTeacherShots(userText, 2);
                NSString *offlineJson = messagesJson;
                if (teacherShots.count > 0 && payload.count >= 2) {
                    NSMutableArray<NSDictionary *> *distilledPayload = [NSMutableArray arrayWithCapacity:(payload.count + teacherShots.count)];
                    [distilledPayload addObject:payload[0]];
                    [distilledPayload addObjectsFromArray:teacherShots];
                    for (NSUInteger pi = 1; pi < payload.count; pi++) {
                        [distilledPayload addObject:payload[pi]];
                    }
                    NSData *dj = [NSJSONSerialization dataWithJSONObject:distilledPayload options:0 error:nil];
                    if (dj.length > 0) {
                        offlineJson = [[NSString alloc] initWithData:dj encoding:NSUTF8StringEncoding] ?: messagesJson;
                        kronos_session_reset(self.session);
                    }
                }
                double pf = 0.0, dc = 0.0;
                size_t pft = 0;
                kronos_session_generate_chat_json(
                    self.session,
                    [offlineJson UTF8String],
                    512,
                    temp,
                    kronos_chat_stream_cb,
                    &sctx,
                    &pf,
                    &dc,
                    &pft
                );
                finalText = [acc copy];
            }
        }

        finalText = kronosVerifyAndRepairIdentityReply(finalText, userText);
        BOOL blockedDigits = kronosContainsUngroundedPlaceholderDigits(finalText, finalUserPrompt, userText);
        if (blockedDigits) {
            finalText = @"I checked your local context and do not have that verified number stored on your device yet, so I will not guess a placeholder value.";
        } else if (usedMac14B && !isAgentTurn && finalText.length > 0) {
            kronosRecordTeacherDistillationPair(userText, finalText);
            kronosSyncContinualMemoryAndDistillVaultWithMac();
        }
        if (finalText.length == 0) {
            finalText = atomic_load(&_cancelFlag) ? @"Stopped." : @"Ready whenever you are.";
        }

        NSString *createdFile = nil;
        if (isAgentTurn) {
            NSArray<NSString *> *afterFiles = [[NSFileManager defaultManager] contentsOfDirectoryAtPath:wsDir error:nil] ?: @[];
            for (NSString *f in afterFiles) {
                if (![f hasPrefix:@"."] && ![beforeFiles containsObject:f]) {
                    createdFile = f;
                    break;
                }
            }
            if (createdFile == nil) {
                createdFile = kronosExtractFallbackFileIfNeeded(userText, finalText);
            }
        }

        dispatch_async(dispatch_get_main_queue(), ^{
            replyBody.attributedText = kronosFormatMessageText(finalText, NO);
            NSString *badge = nil;
            if (createdFile.length > 0) {
                badge = [NSString stringWithFormat:@"Saved %@", createdFile];
                [self addActionPillWithTitle:badge previewFile:createdFile toStack:replyExtra];
                [self refreshWorkspaceFileCount];
            }
            [convMsgs addObject:@{
                @"role": @"assistant",
                @"content": finalText,
                @"displayText": finalText,
                @"actionBadge": badge ?: @"",
                @"previewFile": createdFile ?: @"",
            }];
            self.activeConversation[@"updatedAt"] = @([[NSDate date] timeIntervalSince1970]);
            [KronosConversationStore saveConversations:self.conversations];

            self.isGenerating = NO;
            self.headerSubtitleLabel.text = @"";
            [self updateComposerControls];
            if (self.voiceController.voiceOutputEnabled || self.lastTurnWasSpoken) {
                [self.voiceController speakText:finalText];
            }
            [self scrollToBottomAnimated:YES];
        });
    });
}

@end

@interface KronosSceneDelegate : UIResponder <UIWindowSceneDelegate>
@property (nonatomic, strong) UIWindow *window;
@end

@implementation KronosSceneDelegate
- (void)scene:(UIScene *)scene willConnectToSession:(UISceneSession *)session options:(UISceneConnectionOptions *)connectionOptions {
    (void)session;
    (void)connectionOptions;
    if (![scene isKindOfClass:[UIWindowScene class]]) return;
    UIWindowScene *ws = (UIWindowScene *)scene;
    self.window = [[UIWindow alloc] initWithWindowScene:ws];
    self.window.rootViewController = [[KronosChatViewController alloc] init];
    [self.window makeKeyAndVisible];
}
@end

@interface KronosAppDelegate : UIResponder <UIApplicationDelegate>
@end

@implementation KronosAppDelegate
- (UISceneConfiguration *)application:(UIApplication *)application
configurationForConnectingSceneSession:(UISceneSession *)connectingSceneSession
                              options:(UISceneConnectionOptions *)options {
    (void)application;
    (void)options;
    UISceneConfiguration *cfg = [[UISceneConfiguration alloc] initWithName:@"Default Configuration"
                                                               sessionRole:connectingSceneSession.role];
    cfg.delegateClass = [KronosSceneDelegate class];
    return cfg;
}
@end

int main(int argc, char *argv[]) {
    @autoreleasepool {
        return UIApplicationMain(argc, argv, nil, NSStringFromClass([KronosAppDelegate class]));
    }
}
