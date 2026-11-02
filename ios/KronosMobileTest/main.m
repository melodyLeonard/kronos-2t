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
/// Document hits offered as pills, by pill identifier; the term to retry after a folder grant.
@property (nonatomic, strong) NSMutableDictionary<NSString *, NSDictionary *> *docPillTargets;
@property (nonatomic, copy) NSString *pendingFindTerm;
@property (nonatomic, assign) BOOL pickingFolderForIndex;
@property (nonatomic, assign) BOOL lastTurnWasAppAsk;
@property (nonatomic, copy) NSString *lastAskedApp;
/// Recording strip above the composer: discard, timer, live waveform, stop.
@property (nonatomic, strong) UIView *voiceBar;
@property (nonatomic, strong) NSLayoutConstraint *voiceBarHeight;
@property (nonatomic, strong) UILabel *voiceTimerLabel;
@property (nonatomic, strong) KronosLevelWaveView *voiceWave;
@property (nonatomic, strong) NSTimer *voiceClock;
@property (nonatomic, assign) NSTimeInterval voiceStartedAt;
@property (nonatomic, strong) UIScrollView *attachmentsStrip;
@property (nonatomic, strong) UIStackView *attachmentsStack;
@property (nonatomic, strong) NSLayoutConstraint *attachmentsHeightConstraint;
@property (nonatomic, strong) UIButton *attachButton;
@property (nonatomic, strong) KronosInputTextView *inputTextView;
@property (nonatomic, strong) UILabel *placeholderLabel;
@property (nonatomic, strong) NSLayoutConstraint *inputHeightConstraint;
@property (nonatomic, strong) UIButton *micButton;
@property (nonatomic, strong) UIButton *sendButton;
/// The bubble speaker button whose text is being read aloud.
@property (nonatomic, weak) KronosTapButton *activeSpeakButton;

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

    self.sidebarButton = [KronosTapButton tapButton];
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

    self.ttsToggleButton = [KronosTapButton tapButton];
    self.ttsToggleButton.translatesAutoresizingMaskIntoConstraints = NO;
    UIImageSymbolConfiguration *spkCfg = [UIImageSymbolConfiguration configurationWithPointSize:15.0 weight:UIImageSymbolWeightMedium];
    [self.ttsToggleButton setImage:[UIImage systemImageNamed:@"speaker.slash" withConfiguration:spkCfg] forState:UIControlStateNormal];
    self.ttsToggleButton.tintColor = kronosMutedColor();
    self.ttsToggleButton.accessibilityLabel = @"Toggle spoken voice replies";
    [self.ttsToggleButton addTarget:self action:@selector(ttsToggleTapped) forControlEvents:UIControlEventTouchUpInside];
    [bar addSubview:self.ttsToggleButton];

    self.settingsButton = [KronosTapButton tapButton];
    self.settingsButton.translatesAutoresizingMaskIntoConstraints = NO;
    UIImageSymbolConfiguration *gearCfg = [UIImageSymbolConfiguration configurationWithPointSize:15.5 weight:UIImageSymbolWeightMedium];
    [self.settingsButton setImage:[UIImage systemImageNamed:@"gearshape" withConfiguration:gearCfg] forState:UIControlStateNormal];
    self.settingsButton.tintColor = kronosMutedColor();
    self.settingsButton.accessibilityLabel = @"Settings and Device Context";
    [self.settingsButton addTarget:self action:@selector(openSettingsTapped) forControlEvents:UIControlEventTouchUpInside];
    [bar addSubview:self.settingsButton];

    self.plusNewChatButton = [KronosTapButton tapButton];
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

    self.voiceBar = [[UIView alloc] init];
    self.voiceBar.translatesAutoresizingMaskIntoConstraints = NO;
    self.voiceBar.backgroundColor = kronosSubtleColor();
    self.voiceBar.layer.cornerRadius = 16.0;
    self.voiceBar.clipsToBounds = YES;
    self.voiceBar.hidden = YES;
    [self.composerContainer addSubview:self.voiceBar];

    KronosTapButton *voiceTrash = [KronosTapButton tapButton];
    voiceTrash.translatesAutoresizingMaskIntoConstraints = NO;
    [voiceTrash setImage:[UIImage systemImageNamed:@"trash" withConfiguration:[UIImageSymbolConfiguration configurationWithPointSize:18.0 weight:UIImageSymbolWeightRegular]] forState:UIControlStateNormal];
    voiceTrash.tintColor = kronosMutedColor();
    voiceTrash.accessibilityLabel = @"Discard recording";
    [voiceTrash addTarget:self action:@selector(voiceDiscardTapped) forControlEvents:UIControlEventTouchUpInside];
    [self.voiceBar addSubview:voiceTrash];

    self.voiceTimerLabel = [[UILabel alloc] init];
    self.voiceTimerLabel.translatesAutoresizingMaskIntoConstraints = NO;
    self.voiceTimerLabel.text = @"0:00";
    self.voiceTimerLabel.textColor = kronosTextColor();
    self.voiceTimerLabel.font = [UIFont monospacedDigitSystemFontOfSize:15.0 weight:UIFontWeightMedium];
    [self.voiceBar addSubview:self.voiceTimerLabel];

    self.voiceWave = [[KronosLevelWaveView alloc] init];
    self.voiceWave.translatesAutoresizingMaskIntoConstraints = NO;
    self.voiceWave.tintColor = kronosAccentColor();
    [self.voiceBar addSubview:self.voiceWave];

    KronosTapButton *voiceStop = [KronosTapButton tapButton];
    voiceStop.translatesAutoresizingMaskIntoConstraints = NO;
    [voiceStop setImage:[UIImage systemImageNamed:@"stop.circle.fill" withConfiguration:[UIImageSymbolConfiguration configurationWithPointSize:26.0 weight:UIImageSymbolWeightRegular]] forState:UIControlStateNormal];
    voiceStop.tintColor = [UIColor colorWithRed:0.88 green:0.26 blue:0.28 alpha:1.0];
    voiceStop.accessibilityLabel = @"Stop recording and keep the text";
    [voiceStop addTarget:self action:@selector(voiceStopTapped) forControlEvents:UIControlEventTouchUpInside];
    [self.voiceBar addSubview:voiceStop];

    self.voiceBarHeight = [self.voiceBar.heightAnchor constraintEqualToConstant:0.0];
    [NSLayoutConstraint activateConstraints:@[
        self.voiceBarHeight,
        [voiceTrash.leadingAnchor constraintEqualToAnchor:self.voiceBar.leadingAnchor constant:10.0],
        [voiceTrash.centerYAnchor constraintEqualToAnchor:self.voiceBar.centerYAnchor],
        [voiceTrash.widthAnchor constraintEqualToConstant:36.0],
        [voiceTrash.heightAnchor constraintEqualToConstant:36.0],
        [self.voiceTimerLabel.leadingAnchor constraintEqualToAnchor:voiceTrash.trailingAnchor constant:6.0],
        [self.voiceTimerLabel.centerYAnchor constraintEqualToAnchor:self.voiceBar.centerYAnchor],
        [self.voiceWave.leadingAnchor constraintEqualToAnchor:self.voiceTimerLabel.trailingAnchor constant:10.0],
        [self.voiceWave.trailingAnchor constraintEqualToAnchor:voiceStop.leadingAnchor constant:-10.0],
        [self.voiceWave.centerYAnchor constraintEqualToAnchor:self.voiceBar.centerYAnchor],
        [self.voiceWave.heightAnchor constraintEqualToConstant:30.0],
        [voiceStop.trailingAnchor constraintEqualToAnchor:self.voiceBar.trailingAnchor constant:-8.0],
        [voiceStop.centerYAnchor constraintEqualToAnchor:self.voiceBar.centerYAnchor],
        [voiceStop.widthAnchor constraintEqualToConstant:40.0],
        [voiceStop.heightAnchor constraintEqualToConstant:40.0],
    ]];

    UIView *capsule = [[UIView alloc] init];
    capsule.translatesAutoresizingMaskIntoConstraints = NO;
    capsule.backgroundColor = kronosPanelColor();
    capsule.layer.cornerRadius = 23.0;
    capsule.layer.borderWidth = 1.0;
    capsule.layer.borderColor = kronosLineColor().CGColor;
    [self.composerContainer addSubview:capsule];

    UIImageSymbolConfiguration *iconCfg = [UIImageSymbolConfiguration configurationWithPointSize:18.0 weight:UIImageSymbolWeightMedium];

    self.attachButton = [KronosTapButton tapButton];
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

    self.micButton = [KronosTapButton tapButton];
    self.micButton.translatesAutoresizingMaskIntoConstraints = NO;
    [self.micButton setImage:[UIImage systemImageNamed:@"mic.fill" withConfiguration:iconCfg] forState:UIControlStateNormal];
    self.micButton.tintColor = kronosMutedColor();
    self.micButton.layer.cornerRadius = 17.0;
    self.micButton.accessibilityLabel = @"Speak to Kronos";
    [self.micButton addTarget:self action:@selector(micTapped) forControlEvents:UIControlEventTouchUpInside];
    [capsule addSubview:self.micButton];

    UIImageSymbolConfiguration *sendCfg = [UIImageSymbolConfiguration configurationWithPointSize:28.0 weight:UIImageSymbolWeightSemibold];
    self.sendButton = [KronosTapButton tapButton];
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

        [self.voiceBar.topAnchor constraintEqualToAnchor:self.attachmentsStrip.bottomAnchor constant:6.0],
        [self.voiceBar.leadingAnchor constraintEqualToAnchor:safe.leadingAnchor constant:12.0],
        [self.voiceBar.trailingAnchor constraintEqualToAnchor:safe.trailingAnchor constant:-12.0],
        [capsule.topAnchor constraintEqualToAnchor:self.voiceBar.bottomAnchor constant:6.0],
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
    brandLbl.font = [UIFont systemFontOfSize:22.0 weight:UIFontWeightBold];
    [self.sidebarPanel addSubview:brandLbl];

    UIButton *closeBtn = [KronosTapButton tapButton];
    closeBtn.translatesAutoresizingMaskIntoConstraints = NO;
    UIImageSymbolConfiguration *cfg = [UIImageSymbolConfiguration configurationWithPointSize:16.5 weight:UIImageSymbolWeightMedium];
    [closeBtn setImage:[UIImage systemImageNamed:@"sidebar.left" withConfiguration:cfg] forState:UIControlStateNormal];
    closeBtn.tintColor = kronosSidebarMuted();
    closeBtn.accessibilityLabel = @"Close sidebar";
    [closeBtn addTarget:self action:@selector(closeSidebarAnimated) forControlEvents:UIControlEventTouchUpInside];
    [self.sidebarPanel addSubview:closeBtn];

    UIButton *newConvBtn = [KronosTapButton tapButton];
    newConvBtn.translatesAutoresizingMaskIntoConstraints = NO;
    [newConvBtn setTitle:@"  New conversation" forState:UIControlStateNormal];
    [newConvBtn setImage:[UIImage systemImageNamed:@"plus" withConfiguration:cfg] forState:UIControlStateNormal];
    newConvBtn.tintColor = kronosSidebarText();
    [newConvBtn setTitleColor:kronosSidebarText() forState:UIControlStateNormal];
    newConvBtn.titleLabel.font = [UIFont systemFontOfSize:16.0 weight:UIFontWeightSemibold];
    newConvBtn.contentHorizontalAlignment = UIControlContentHorizontalAlignmentLeading;
    newConvBtn.contentEdgeInsets = UIEdgeInsetsMake(0, 14, 0, 14);
    newConvBtn.backgroundColor = kronosSidebarPanel();
    newConvBtn.layer.cornerRadius = 14.0;
    newConvBtn.layer.borderWidth = 1.0;
    newConvBtn.layer.borderColor = [UIColor colorWithWhite:1.0 alpha:0.09].CGColor;
    [newConvBtn addTarget:self action:@selector(newConversationFromSidebar) forControlEvents:UIControlEventTouchUpInside];
    [self.sidebarPanel addSubview:newConvBtn];

    UIView *searchBox = [[UIView alloc] init];
    searchBox.translatesAutoresizingMaskIntoConstraints = NO;
    searchBox.backgroundColor = kronosSidebarPanel();
    searchBox.layer.cornerRadius = 12.0;
    [self.sidebarPanel addSubview:searchBox];

    UIImageView *mag = [[UIImageView alloc] initWithImage:[UIImage systemImageNamed:@"magnifyingglass"]];
    mag.translatesAutoresizingMaskIntoConstraints = NO;
    mag.tintColor = kronosSidebarMuted();
    [searchBox addSubview:mag];

    self.sidebarSearchField = [[UITextField alloc] init];
    self.sidebarSearchField.translatesAutoresizingMaskIntoConstraints = NO;
    self.sidebarSearchField.textColor = kronosSidebarText();
    self.sidebarSearchField.font = [UIFont systemFontOfSize:15.0 weight:UIFontWeightRegular];
    self.sidebarSearchField.attributedPlaceholder = [[NSAttributedString alloc] initWithString:@"Search conversations..."
                                                                                    attributes:@{NSForegroundColorAttributeName: kronosSidebarMuted()}];
    self.sidebarSearchField.clearButtonMode = UITextFieldViewModeWhileEditing;
    self.sidebarSearchField.delegate = self;
    [self.sidebarSearchField addTarget:self action:@selector(searchFieldChanged:) forControlEvents:UIControlEventEditingChanged];
    [searchBox addSubview:self.sidebarSearchField];

    UIButton *filesRow = [KronosTapButton tapButton];
    filesRow.translatesAutoresizingMaskIntoConstraints = NO;
    [filesRow setTitle:@"  Workspace files" forState:UIControlStateNormal];
    [filesRow setImage:[UIImage systemImageNamed:@"folder"] forState:UIControlStateNormal];
    filesRow.tintColor = kronosAccentColor();
    [filesRow setTitleColor:kronosSidebarText() forState:UIControlStateNormal];
    filesRow.titleLabel.font = [UIFont systemFontOfSize:15.5 weight:UIFontWeightMedium];
    filesRow.contentHorizontalAlignment = UIControlContentHorizontalAlignmentLeading;
    filesRow.contentEdgeInsets = UIEdgeInsetsMake(0, 14, 0, 8);
    filesRow.layer.cornerRadius = 12.0;
    [filesRow addTarget:self action:@selector(openWorkspaceFilesTapped) forControlEvents:UIControlEventTouchUpInside];
    [self.sidebarPanel addSubview:filesRow];

    self.workspaceCountBadge = [[UILabel alloc] init];
    self.workspaceCountBadge.translatesAutoresizingMaskIntoConstraints = NO;
    self.workspaceCountBadge.text = @"0";
    self.workspaceCountBadge.textColor = kronosSidebarMuted();
    self.workspaceCountBadge.font = [UIFont monospacedSystemFontOfSize:12.0 weight:UIFontWeightSemibold];
    self.workspaceCountBadge.textAlignment = NSTextAlignmentCenter;
    self.workspaceCountBadge.backgroundColor = kronosSidebarPanel();
    self.workspaceCountBadge.layer.cornerRadius = 11.0;
    self.workspaceCountBadge.clipsToBounds = YES;
    [self.sidebarPanel addSubview:self.workspaceCountBadge];

    UILabel *sectionHdr = [[UILabel alloc] init];
    sectionHdr.translatesAutoresizingMaskIntoConstraints = NO;
    sectionHdr.attributedText = [[NSAttributedString alloc] initWithString:@"RECENT CONVERSATIONS" attributes:@{NSKernAttributeName: @0.6}];
    sectionHdr.textColor = kronosSidebarMuted();
    sectionHdr.font = [UIFont systemFontOfSize:12.0 weight:UIFontWeightSemibold];
    [self.sidebarPanel addSubview:sectionHdr];

    self.conversationsTable = [[UITableView alloc] initWithFrame:CGRectZero style:UITableViewStylePlain];
    self.conversationsTable.translatesAutoresizingMaskIntoConstraints = NO;
    self.conversationsTable.backgroundColor = [UIColor clearColor];
    self.conversationsTable.separatorStyle = UITableViewCellSeparatorStyleNone;
    self.conversationsTable.rowHeight = 64.0;
    self.conversationsTable.contentInset = UIEdgeInsetsMake(0, 0, 12, 0);
    self.conversationsTable.dataSource = self;
    self.conversationsTable.delegate = self;
    [self.sidebarPanel addSubview:self.conversationsTable];

    UIControl *settingsFooter = [[KronosPressControl alloc] init];
    settingsFooter.translatesAutoresizingMaskIntoConstraints = NO;
    settingsFooter.backgroundColor = kronosSidebarPanel();
    settingsFooter.layer.cornerRadius = 14.0;
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
    footerTitle.text = @"Settings";
    footerTitle.textColor = kronosSidebarText();
    footerTitle.font = [UIFont systemFontOfSize:15.0 weight:UIFontWeightSemibold];
    [settingsFooter addSubview:footerTitle];

    self.sidebarSettingsSubtitle = [[UILabel alloc] init];
    self.sidebarSettingsSubtitle.translatesAutoresizingMaskIntoConstraints = NO;
    self.sidebarSettingsSubtitle.text = @"Context on · Phone on";
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

        [brandLbl.topAnchor constraintEqualToAnchor:safe.topAnchor constant:18.0],
        [brandLbl.leadingAnchor constraintEqualToAnchor:self.sidebarPanel.leadingAnchor constant:20.0],

        [closeBtn.centerYAnchor constraintEqualToAnchor:brandLbl.centerYAnchor],
        [closeBtn.trailingAnchor constraintEqualToAnchor:self.sidebarPanel.trailingAnchor constant:-14.0],
        [closeBtn.widthAnchor constraintEqualToConstant:40.0],
        [closeBtn.heightAnchor constraintEqualToConstant:40.0],

        [newConvBtn.topAnchor constraintEqualToAnchor:brandLbl.bottomAnchor constant:20.0],
        [newConvBtn.leadingAnchor constraintEqualToAnchor:self.sidebarPanel.leadingAnchor constant:16.0],
        [newConvBtn.trailingAnchor constraintEqualToAnchor:self.sidebarPanel.trailingAnchor constant:-16.0],
        [newConvBtn.heightAnchor constraintEqualToConstant:50.0],

        [searchBox.topAnchor constraintEqualToAnchor:newConvBtn.bottomAnchor constant:12.0],
        [searchBox.leadingAnchor constraintEqualToAnchor:newConvBtn.leadingAnchor],
        [searchBox.trailingAnchor constraintEqualToAnchor:newConvBtn.trailingAnchor],
        [searchBox.heightAnchor constraintEqualToConstant:44.0],

        [mag.leadingAnchor constraintEqualToAnchor:searchBox.leadingAnchor constant:14.0],
        [mag.centerYAnchor constraintEqualToAnchor:searchBox.centerYAnchor],
        [mag.widthAnchor constraintEqualToConstant:17.0],
        [mag.heightAnchor constraintEqualToConstant:17.0],

        [self.sidebarSearchField.leadingAnchor constraintEqualToAnchor:mag.trailingAnchor constant:10.0],
        [self.sidebarSearchField.trailingAnchor constraintEqualToAnchor:searchBox.trailingAnchor constant:-12.0],
        [self.sidebarSearchField.centerYAnchor constraintEqualToAnchor:searchBox.centerYAnchor],

        [filesRow.topAnchor constraintEqualToAnchor:searchBox.bottomAnchor constant:14.0],
        [filesRow.leadingAnchor constraintEqualToAnchor:self.sidebarPanel.leadingAnchor constant:16.0],
        [filesRow.trailingAnchor constraintEqualToAnchor:self.workspaceCountBadge.leadingAnchor constant:-8.0],
        [filesRow.heightAnchor constraintEqualToConstant:46.0],

        [self.workspaceCountBadge.centerYAnchor constraintEqualToAnchor:filesRow.centerYAnchor],
        [self.workspaceCountBadge.trailingAnchor constraintEqualToAnchor:self.sidebarPanel.trailingAnchor constant:-20.0],
        [self.workspaceCountBadge.heightAnchor constraintEqualToConstant:22.0],
        [self.workspaceCountBadge.widthAnchor constraintGreaterThanOrEqualToConstant:30.0],

        [sectionHdr.topAnchor constraintEqualToAnchor:filesRow.bottomAnchor constant:22.0],
        [sectionHdr.leadingAnchor constraintEqualToAnchor:self.sidebarPanel.leadingAnchor constant:22.0],

        [self.conversationsTable.topAnchor constraintEqualToAnchor:sectionHdr.bottomAnchor constant:10.0],
        [self.conversationsTable.leadingAnchor constraintEqualToAnchor:self.sidebarPanel.leadingAnchor constant:12.0],
        [self.conversationsTable.trailingAnchor constraintEqualToAnchor:self.sidebarPanel.trailingAnchor constant:-12.0],
        [self.conversationsTable.bottomAnchor constraintEqualToAnchor:settingsFooter.topAnchor constant:-12.0],

        [settingsFooter.leadingAnchor constraintEqualToAnchor:self.sidebarPanel.leadingAnchor constant:16.0],
        [settingsFooter.trailingAnchor constraintEqualToAnchor:self.sidebarPanel.trailingAnchor constant:-16.0],
        [settingsFooter.bottomAnchor constraintEqualToAnchor:safe.bottomAnchor constant:-16.0],
        [settingsFooter.heightAnchor constraintEqualToConstant:62.0],

        [footerGear.leadingAnchor constraintEqualToAnchor:settingsFooter.leadingAnchor constant:16.0],
        [footerGear.centerYAnchor constraintEqualToAnchor:settingsFooter.centerYAnchor],
        [footerGear.widthAnchor constraintEqualToConstant:22.0],
        [footerGear.heightAnchor constraintEqualToConstant:22.0],

        [footerTitle.topAnchor constraintEqualToAnchor:settingsFooter.topAnchor constant:12.0],
        [footerTitle.leadingAnchor constraintEqualToAnchor:footerGear.trailingAnchor constant:12.0],
        [footerTitle.trailingAnchor constraintEqualToAnchor:footerChev.leadingAnchor constant:-6.0],

        [self.sidebarSettingsSubtitle.topAnchor constraintEqualToAnchor:footerTitle.bottomAnchor constant:2.0],
        [self.sidebarSettingsSubtitle.leadingAnchor constraintEqualToAnchor:footerTitle.leadingAnchor],
        [self.sidebarSettingsSubtitle.trailingAnchor constraintEqualToAnchor:footerTitle.trailingAnchor],

        [footerChev.trailingAnchor constraintEqualToAnchor:settingsFooter.trailingAnchor constant:-16.0],
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
    NSString *ctxPart = s.developDeviceContext ? @"Context on" : @"Context off";
    NSString *phonePart = s.allowPhoneUse ? @"Phone on" : @"Phone off";
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
        cell.selectionStyle = UITableViewCellSelectionStyleNone;
        // Rounded row card with a 3 pt gap to its neighbours.
        cell.contentView.layer.cornerRadius = 12.0;
        cell.contentView.clipsToBounds = YES;
        cell.contentView.directionalLayoutMargins = NSDirectionalEdgeInsetsMake(10, 14, 10, 14);
        cell.preservesSuperviewLayoutMargins = NO;
        cell.contentView.preservesSuperviewLayoutMargins = NO;
        cell.textLabel.numberOfLines = 1;
    }
    cell.contentView.frame = CGRectInset(cell.bounds, 0, 3);
    NSDictionary *conv = self.filteredConversations[(NSUInteger)indexPath.row];
    BOOL isActive = [conv[@"id"] isEqual:self.activeConversation[@"id"]];
    cell.contentView.backgroundColor = isActive ? kronosSidebarActive() : [UIColor clearColor];
    cell.textLabel.text = conv[@"title"] ?: @"New conversation";
    cell.textLabel.textColor = kronosSidebarText();
    cell.textLabel.font = [UIFont systemFontOfSize:15.5 weight:isActive ? UIFontWeightSemibold : UIFontWeightMedium];
    NSTimeInterval ts = [conv[@"updatedAt"] doubleValue];
    cell.detailTextLabel.text = kronosRelativeDateString(ts);
    cell.detailTextLabel.textColor = kronosSidebarMuted();
    cell.detailTextLabel.font = [UIFont systemFontOfSize:12.5 weight:UIFontWeightRegular];
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
            body.attributedText = kronosFormatMessageText(kronosStripThinkBlocks(content), NO);

            if (previewImageFile.length > 0) {
                NSString *imgPath = [kronosWorkspaceDirectory() stringByAppendingPathComponent:previewImageFile];
                UIImage *savedImg = [UIImage imageWithContentsOfFile:imgPath];
                if (savedImg != nil) {
                    [self addPhotoPreviewCardWithImage:savedImg caption:@"Selected photo from Photo Library" toStack:extra];
                }
            }
            if ([m[@"waThread"] isKindOfClass:[NSArray class]]) [self addWhatsAppThread:m[@"waThread"] toStack:extra];
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
        UIButton *card = [KronosTapButton tapButton];
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
    if (self.pickingFolderForIndex) {
        self.pickingFolderForIndex = NO;
        KronosIosDeviceContextManager *mgr = [KronosIosDeviceContextManager sharedManager];
        BOOL added = NO;
        for (NSURL *u in urls) added = [mgr addIndexedFolderURL:u] || added;
        if (!added) kronosShowToast(self.view, @"Choose iCloud Drive, not Kronos's folder");
        NSString *term = self.pendingFindTerm;
        self.pendingFindTerm = nil;
        self.headerSubtitleLabel.text = @"Indexing documents…";
        dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
            [mgr reindexDocuments];
            dispatch_async(dispatch_get_main_queue(), ^{
                self.headerSubtitleLabel.text = @"";
                [[KronosIosDeviceContextManager sharedManager] refreshDeviceContextInteractive:NO completion:nil];
                if (term.length > 0) {
                    UILabel *replyBody = nil;
                    UIStackView *replyExtra = nil;
                    [self appendKronosBubbleOutBody:&replyBody outExtraStack:&replyExtra];
                    [self answerFindRequest:term replyBody:replyBody replyExtra:replyExtra convMsgs:self.activeConversation[@"messages"]];
                    [self scrollToBottomAnimated:YES];
                }
            });
        });
        return;
    }
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

        UIButton *rm = [KronosTapButton tapButton];
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
    [[[UISelectionFeedbackGenerator alloc] init] selectionChanged];
    if ([self.ttsToggleButton isKindOfClass:[KronosTapButton class]]) [(KronosTapButton *)self.ttsToggleButton bounce];
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
    KronosTapButton *mic = [self.micButton isKindOfClass:[KronosTapButton class]] ? (KronosTapButton *)self.micButton : nil;
    [self.voiceClock invalidate];
    self.voiceClock = nil;
    if (isListening) {
        self.voiceStartedAt = [NSDate timeIntervalSinceReferenceDate];
        self.voiceTimerLabel.text = @"0:00";
        [self.voiceWave reset];
        __weak typeof(self) weakSelf = self;
        self.voiceClock = [NSTimer scheduledTimerWithTimeInterval:0.5 repeats:YES block:^(NSTimer *t) {
            __strong typeof(weakSelf) me = weakSelf;
            if (me == nil) return;
            NSInteger sec = (NSInteger)([NSDate timeIntervalSinceReferenceDate] - me.voiceStartedAt);
            me.voiceTimerLabel.text = [NSString stringWithFormat:@"%ld:%02ld", (long)(sec / 60), (long)(sec % 60)];
        }];
    }
    BOOL showBar = isListening;
    if (self.voiceBar.hidden == showBar) {
        self.voiceBar.hidden = NO;
        [UIView animateWithDuration:0.22 delay:0 options:UIViewAnimationOptionCurveEaseOut animations:^{
            self.voiceBarHeight.constant = showBar ? 52.0 : 0.0;
            self.voiceBar.alpha = showBar ? 1.0 : 0.0;
            [self.view layoutIfNeeded];
        } completion:^(BOOL f) { self.voiceBar.hidden = !showBar; }];
    }
    if (isListening) {
        [self.micButton setImage:[UIImage systemImageNamed:@"waveform" withConfiguration:iconCfg] forState:UIControlStateNormal];
        self.micButton.tintColor = [UIColor whiteColor];
        self.micButton.backgroundColor = [UIColor colorWithRed:0.88 green:0.26 blue:0.28 alpha:1.0];
        self.headerSubtitleLabel.text = status ?: @"Listening...";
        mic.pulsing = YES;
    } else {
        mic.pulsing = NO;
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

/// The transcript stays in the composer; nothing is sent until the user presses Send.
- (void)voiceControllerDidFinishUtterance:(NSString *)finalTranscript {
    self.inputTextView.text = finalTranscript ?: @"";
    [self textViewDidChange:self.inputTextView];
    if (finalTranscript.length > 0) {
        self.lastTurnWasSpoken = YES;
        if ([self.sendButton isKindOfClass:[KronosTapButton class]]) [(KronosTapButton *)self.sendButton bounce];
    }
}

- (void)voiceControllerDidUpdateLevel:(float)level {
    [self.voiceWave pushLevel:level];
}

- (void)voiceDiscardTapped {
    [self.voiceController discardListening];
    self.inputTextView.text = @"";
    [self textViewDidChange:self.inputTextView];
}

- (void)voiceStopTapped {
    [self.voiceController stopListeningAndCommit:YES];
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
            kronos_session_set_thinking(sess, 0);
#if !TARGET_OS_SIMULATOR
            kronos_session_enable_metal(sess);
#endif
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
        // Falls through: the final transcript is now in the composer and Send means send.
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

    UIButton *speakBtn = [KronosTapButton tapButton];
    speakBtn.translatesAutoresizingMaskIntoConstraints = NO;
    [speakBtn setImage:[UIImage systemImageNamed:@"speaker.wave.2"] forState:UIControlStateNormal];
    speakBtn.tintColor = kronosMutedColor();
    [speakBtn addTarget:self action:@selector(speakBubbleTapped:) forControlEvents:UIControlEventTouchUpInside];
    [hdr addSubview:speakBtn];

    UIButton *copyBtn = [KronosTapButton tapButton];
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
    UIButton *pill = [KronosTapButton tapButton];
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

/// Searches the workspace, this conversation's attachments and the granted-folder document index;
/// opens the single best hit, lists several, or offers a folder grant / the pickers.
- (void)answerFindRequest:(NSString *)findTerm replyBody:(UILabel *)replyBody replyExtra:(UIStackView *)replyExtra convMsgs:(NSMutableArray *)convMsgs {
    KronosIosDeviceContextManager *mgr = [KronosIosDeviceContextManager sharedManager];
    NSArray<NSString *> *wsHits = [self searchReadableFilesForTerm:findTerm];
    NSArray<NSDictionary *> *docHits = [mgr searchDocuments:findTerm limit:6];
    NSUInteger total = wsHits.count + docHits.count;
    if (total > 0) {
        NSString *msg;
        if (total == 1) {
            NSString *name = wsHits.count > 0 ? wsHits.firstObject : docHits.firstObject[@"name"];
            NSString *where = wsHits.count > 0 ? @"the Kronos workspace" : docHits.firstObject[@"root"];
            msg = [NSString stringWithFormat:@"Found \"%@\" in %@. Opening it.", name, where];
        } else {
            msg = [NSString stringWithFormat:@"Found %lu documents matching \"%@\" on this iPhone. Tap one to open it.", (unsigned long)total, findTerm];
        }
        [self recordImmediateKronosReply:msg actionBadge:nil replyBody:replyBody replyExtra:replyExtra convMsgs:convMsgs];
        for (NSString *f in wsHits) [self addActionPillWithTitle:f previewFile:f toStack:replyExtra];
        if (self.docPillTargets == nil) self.docPillTargets = [NSMutableDictionary dictionary];
        for (NSDictionary *d in docHits) {
            NSString *key = [NSString stringWithFormat:@"doc:%@/%@", d[@"folder"], d[@"rel"]];
            self.docPillTargets[key] = d;
            NSString *title = [NSString stringWithFormat:@"%@ · %@%@", d[@"name"], d[@"root"], [d[@"placeholder"] boolValue] ? @" (iCloud)" : @""];
            UIButton *pill = [KronosTapButton tapButton];
            pill.translatesAutoresizingMaskIntoConstraints = NO;
            [pill setImage:[UIImage systemImageNamed:@"doc.richtext"] forState:UIControlStateNormal];
            [pill setTitle:[NSString stringWithFormat:@"  %@", title] forState:UIControlStateNormal];
            pill.tintColor = kronosAccentColor();
            [pill setTitleColor:kronosAccentColor() forState:UIControlStateNormal];
            pill.titleLabel.font = [UIFont systemFontOfSize:13.0 weight:UIFontWeightSemibold];
            pill.titleLabel.lineBreakMode = NSLineBreakByTruncatingMiddle;
            pill.contentHorizontalAlignment = UIControlContentHorizontalAlignmentLeading;
            pill.backgroundColor = [kronosAccentColor() colorWithAlphaComponent:0.12];
            pill.layer.cornerRadius = 10.0;
            [pill.heightAnchor constraintEqualToConstant:36.0].active = YES;
            pill.accessibilityIdentifier = key;
            [pill addTarget:self action:@selector(openIndexedDocumentTapped:) forControlEvents:UIControlEventTouchUpInside];
            [replyExtra addArrangedSubview:pill];
        }
        if (total == 1) {
            if (wsHits.count > 0) [self openFilePreviewForFilename:wsHits.firstObject];
            else [self openIndexedDocument:docHits.firstObject];
        }
        return;
    }
    NSUInteger folders = [mgr indexedFolderCount];
    if (folders == 0) {
        self.pendingFindTerm = findTerm;
        NSString *msg = [NSString stringWithFormat:@"Kronos has not been given a folder to search yet, so it cannot see \"%@\". Tap Grant below, then Browse, iCloud Drive, Open. Kronos then indexes every document there and opens it on request. You only do this once.", findTerm];
        [self recordImmediateKronosReply:msg actionBadge:nil replyBody:replyBody replyExtra:replyExtra convMsgs:convMsgs];
        [self addActionPillWithTitle:@"Grant a folder to search (iCloud Drive)" symbol:@"folder.badge.plus" action:@selector(grantFolderTapped) toStack:replyExtra];
        [self addActionPillWithTitle:@"Choose the file directly" symbol:@"doc" action:@selector(pickDocumentTapped) toStack:replyExtra];
        return;
    }
    NSString *msg = [NSString stringWithFormat:@"No document matching \"%@\" in the %lu folder%@ Kronos indexes (%lu documents) or the workspace. Grant another folder, or pick the file directly.",
                     findTerm, (unsigned long)folders, folders == 1 ? @"" : @"s", (unsigned long)mgr.documentsCount];
    [self recordImmediateKronosReply:msg actionBadge:nil replyBody:replyBody replyExtra:replyExtra convMsgs:convMsgs];
    self.pendingFindTerm = findTerm;
    [self addActionPillWithTitle:@"Grant another folder" symbol:@"folder.badge.plus" action:@selector(grantFolderTapped) toStack:replyExtra];
    [self addActionPillWithTitle:@"Choose from Files" symbol:@"folder" action:@selector(pickDocumentTapped) toStack:replyExtra];
    [self addActionPillWithTitle:@"Choose from Photos" symbol:@"photo.on.rectangle" action:@selector(pickPhotoTapped) toStack:replyExtra];
}

- (void)grantFolderTapped {
    self.pickingFolderForIndex = YES;
    UIDocumentPickerViewController *picker = [[UIDocumentPickerViewController alloc] initForOpeningContentTypes:@[UTTypeFolder]];
    picker.allowsMultipleSelection = YES;
    picker.delegate = self;
    [self presentViewController:picker animated:YES completion:nil];
}

- (void)openIndexedDocumentTapped:(UIButton *)sender {
    NSDictionary *doc = self.docPillTargets[sender.accessibilityIdentifier ?: @""];
    if (doc != nil) [self openIndexedDocument:doc];
}

/// Copies the document into the workspace cache (downloading from iCloud if needed) and previews it.
- (void)openIndexedDocument:(NSDictionary *)doc {
    self.headerSubtitleLabel.text = [doc[@"placeholder"] boolValue] ? @"Downloading from iCloud…" : @"Opening…";
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        NSString *cacheDir = [kronosWorkspaceDirectory() stringByAppendingPathComponent:@".opened"];
        NSString *local = [[KronosIosDeviceContextManager sharedManager] materializeDocument:doc intoDirectory:cacheDir];
        dispatch_async(dispatch_get_main_queue(), ^{
            self.headerSubtitleLabel.text = @"";
            if (local == nil) {
                UIAlertController *a = [UIAlertController alertControllerWithTitle:@"Could not open" message:[NSString stringWithFormat:@"%@ is not available right now (it may still be downloading from iCloud).", doc[@"name"]] preferredStyle:UIAlertControllerStyleAlert];
                [a addAction:[UIAlertAction actionWithTitle:@"OK" style:UIAlertActionStyleDefault handler:nil]];
                [self presentViewController:a animated:YES completion:nil];
                return;
            }
            KronosFilePreviewViewController *pvc = [[KronosFilePreviewViewController alloc] init];
            pvc.filePath = local;
            UINavigationController *nav = [[UINavigationController alloc] initWithRootViewController:pvc];
            [self presentViewController:nav animated:YES completion:nil];
        });
    });
}

- (void)answerLatestMediaKind:(NSInteger)kind countOnly:(BOOL)countOnly replyBody:(UILabel *)replyBody replyExtra:(UIStackView *)replyExtra convMsgs:(NSMutableArray *)convMsgs {
    if (![KronosMobileSettings sharedSettings].developDeviceContext) {
        [self recordImmediateKronosReply:@"Develop Device Context is off, so Kronos is not reading your Photo Library. Turn it on in Settings and ask again." actionBadge:nil replyBody:replyBody replyExtra:replyExtra convMsgs:convMsgs];
        return;
    }
    NSString *noun = kind == 1 ? @"screenshot" : (kind == 2 ? @"video" : @"photo");
    self.headerSubtitleLabel.text = @"Reading Photos…";
    [[KronosIosDeviceContextManager sharedManager] fetchLatestMediaOfKind:kind completion:^(UIImage *image, NSDate *date, NSUInteger total, BOOL limited, NSString *errorNote) {
        if (errorNote != nil || date == nil) {
            self.headerSubtitleLabel.text = @"";
            NSString *msg = errorNote ?: [NSString stringWithFormat:@"No %@s in your Photo Library%@.", noun, limited ? @" among the photos Kronos can see" : @""];
            [self recordImmediateKronosReply:msg actionBadge:nil replyBody:replyBody replyExtra:replyExtra convMsgs:convMsgs];
            if (errorNote != nil) [self addActionPillWithTitle:@"Open iPhone Settings" symbol:@"gear" action:@selector(openAppSettingsTapped) toStack:replyExtra];
            return;
        }
        dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
            NSString *seen = nil;
            if (kind != 2 && !countOnly && image != nil) {
                NSDictionary *rep = kronosAnalyzeUprightImageWithVision(kronosRenderUprightImage(image, 1280.0));
                NSArray<NSString *> *lines = rep[@"ocrLines"];
                NSArray<NSString *> *objects = rep[@"objects"];
                if ([rep[@"ocrChars"] unsignedIntegerValue] >= 35) {
                    NSString *text = [[lines subarrayWithRange:NSMakeRange(0, MIN((NSUInteger)6, lines.count))] componentsJoinedByString:@" "];
                    if (text.length > 220) text = [[text substringToIndex:220] stringByAppendingString:@"…"];
                    seen = [NSString stringWithFormat:@"Text in it: \"%@\"", text];
                } else if (objects.count > 0) {
                    seen = [NSString stringWithFormat:@"It appears to show: %@.", [[objects subarrayWithRange:NSMakeRange(0, MIN((NSUInteger)3, objects.count))] componentsJoinedByString:@", "]];
                }
            }
            NSDateFormatter *stamp = [[NSDateFormatter alloc] init];
            stamp.dateFormat = @"yyyyMMdd-HHmmss";
            NSString *dir = [kronosWorkspaceDirectory() stringByAppendingPathComponent:@".opened"];
            [[NSFileManager defaultManager] createDirectoryAtPath:dir withIntermediateDirectories:YES attributes:nil error:nil];
            NSString *path = [dir stringByAppendingPathComponent:[NSString stringWithFormat:@"%@-%@.jpg", noun, [stamp stringFromDate:date]]];
            NSData *jpeg = (kind != 2 && image != nil) ? UIImageJPEGRepresentation(image, 0.9) : nil;
            BOOL saved = jpeg != nil && [jpeg writeToFile:path atomically:YES];
            dispatch_async(dispatch_get_main_queue(), ^{
                self.headerSubtitleLabel.text = @"";
                NSDateFormatter *df = [[NSDateFormatter alloc] init];
                df.dateStyle = NSDateFormatterLongStyle;
                df.timeStyle = NSDateFormatterShortStyle;
                NSString *when = [df stringFromDate:date];
                NSMutableString *msg = [NSMutableString string];
                if (countOnly) [msg appendFormat:@"You have %lu %@%@ on this iPhone. The latest is from %@.", (unsigned long)total, noun, total == 1 ? @"" : @"s", when];
                else [msg appendFormat:@"Your last %@ was %@ on %@.", noun, kind == 2 ? @"recorded" : @"taken", when];
                if (seen != nil) [msg appendFormat:@" %@", seen];
                if (limited) [msg appendString:@" Kronos only has access to the photos you selected, so a newer one may exist."];
                [self recordImmediateKronosReply:msg actionBadge:nil replyBody:replyBody replyExtra:replyExtra convMsgs:convMsgs];
                if (image != nil) {
                    [self addPhotoPreviewCardWithImage:image caption:when toStack:replyExtra];
                    if (saved) {
                        [self addActionPillWithTitle:[NSString stringWithFormat:@"Open %@", noun] symbol:@"arrow.up.left.and.arrow.down.right" action:@selector(openSavedMediaTapped:) toStack:replyExtra];
                        replyExtra.arrangedSubviews.lastObject.accessibilityIdentifier = path;
                    }
                }
                [self scrollToBottomAnimated:YES];
            });
        });
    }];
}

/// Answers from the real messages without the model, which misquoted them in testing.
/// Returns NO when the context holds no thread lines.
- (BOOL)answerThreadQuestion:(NSString *)question fromContext:(NSString *)context replyBody:(UILabel *)replyBody replyExtra:(UIStackView *)replyExtra convMsgs:(NSMutableArray *)convMsgs {
    NSMutableArray<NSDictionary *> *msgs = [NSMutableArray array];
    NSDateFormatter *dayFmt = [[NSDateFormatter alloc] init];
    dayFmt.dateFormat = @"yyyy-MM-dd";
    NSString *today = [dayFmt stringFromDate:[NSDate date]];
    for (NSString *ln in [context componentsSeparatedByCharactersInSet:[NSCharacterSet newlineCharacterSet]]) {
        NSString *t = [ln stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceCharacterSet]];
        if (![t hasPrefix:@"- ["]) continue;
        NSRange close = [t rangeOfString:@"] "];
        if (close.location == NSNotFound || close.location < 8) continue;
        NSString *stamp = [t substringWithRange:NSMakeRange(3, close.location - 3)];
        if (stamp.length >= 16) stamp = [stamp hasPrefix:today] ? [stamp substringWithRange:NSMakeRange(11, 5)] : [stamp substringToIndex:16];
        NSString *body = [t substringFromIndex:close.location + 2];
        NSRange colon = [body rangeOfString:@": "];
        if (colon.location == NSNotFound) continue;
        BOOL mine = [body hasPrefix:@"You"];
        NSString *who = mine ? @"You" : [[body substringToIndex:colon.location] stringByReplacingOccurrencesOfString:@"From " withString:@""];
        [msgs addObject:@{@"t": stamp, @"who": who, @"text": [body substringFromIndex:colon.location + 2]}];
    }
    if (msgs.count == 0) return NO;

    NSString *contact = @"";
    for (NSDictionary *m in msgs) {
        if (![m[@"who"] isEqualToString:@"You"]) { contact = [m[@"who"] lowercaseString]; break; }
    }
    NSMutableArray<NSString *> *keys = [NSMutableArray array];
    for (NSString *term in kronosExtractExternalLookupTerms(question)) {
        NSString *low = [term lowercaseString];
        if (low.length >= 4 && ![contact containsString:low]) [keys addObject:[low substringToIndex:MIN((NSUInteger)5, low.length)]];
    }
    NSMutableIndexSet *hits = [NSMutableIndexSet indexSet];
    for (NSUInteger i = 0; i < msgs.count && hits.count < 4; i++) { // newest first
        NSString *low = [msgs[i][@"text"] lowercaseString];
        for (NSString *k in keys) {
            if ([low containsString:k]) { [hits addIndex:i]; break; }
        }
    }
    NSDictionary *hit = hits.count > 0 ? msgs[hits.firstIndex] : nil;
    NSString *(^describe)(NSDictionary *) = ^NSString *(NSDictionary *m) {
        NSString *who = [m[@"who"] isEqualToString:@"You"] ? @"you" : m[@"who"];
        return [NSString stringWithFormat:@"from %@ at %@: \"%@\"", who, m[@"t"], m[@"text"]];
    };
    NSDictionary *latestIn = nil, *latestOut = nil;
    for (NSDictionary *m in msgs) {
        if ([m[@"who"] isEqualToString:@"You"]) { if (latestOut == nil) latestOut = m; }
        else if (latestIn == nil) latestIn = m;
    }
    NSString *contactName = latestIn[@"who"] ?: @"them";
    KronosMessageDirectionIntent intent = kronosParseMessageDirectionIntent(question);
    NSString *reply = nil;
    NSRange lead = [context rangeOfString:@"VERIFIED DIRECT ANSWER ("];
    if (lead.location != NSNotFound) {
        NSString *tail = [context substringFromIndex:lead.location + lead.length];
        NSRange end = [tail rangeOfString:@"): "];
        if (end.location != NSNotFound && ([tail hasPrefix:@"Yes,"] || [tail hasPrefix:@"No,"])) {
            reply = [[[tail substringToIndex:end.location] stringByReplacingOccurrencesOfString:@"`" withString:@""] stringByAppendingString:@"."];
        }
    }
    if (reply != nil) {
    } else if (intent == KronosMsgIntentIncomingFromContact && latestIn != nil) {
        reply = [NSString stringWithFormat:@"%@'s last message to you was at %@: \"%@\"", contactName, latestIn[@"t"], latestIn[@"text"]];
    } else if (intent == KronosMsgIntentOutgoingFromUser && latestOut != nil) {
        reply = [NSString stringWithFormat:@"Your last message to %@ was at %@: \"%@\"", contactName, latestOut[@"t"], latestOut[@"text"]];
    } else if (hit != nil) {
        reply = [NSString stringWithFormat:@"The latest message about this is %@", describe(hit)];
    } else if (keys.count > 0) {
        reply = [NSString stringWithFormat:@"Nothing in your recent messages mentions this. The latest message is %@", describe(msgs.firstObject)];
    } else {
        reply = [NSString stringWithFormat:@"The latest message is %@", describe(msgs.firstObject)];
    }
    // Bubbles: the matching messages with one neighbour each, plus the newest from each side.
    NSMutableIndexSet *shown = [NSMutableIndexSet indexSet];
    [hits enumerateIndexesUsingBlock:^(NSUInteger i, BOOL *stop) {
        (void)stop;
        [shown addIndex:i];
        if (i > 0) [shown addIndex:i - 1];
        if (i + 1 < msgs.count) [shown addIndex:i + 1];
    }];
    [shown addIndex:0];
    if (latestIn != nil) [shown addIndex:[msgs indexOfObject:latestIn]];
    if (hits.count == 0) [shown addIndexesInRange:NSMakeRange(0, MIN((NSUInteger)6, msgs.count))];
    NSMutableArray<NSDictionary *> *thread = [NSMutableArray array];
    __block NSUInteger prev = NSNotFound;
    [shown enumerateIndexesWithOptions:NSEnumerationReverse usingBlock:^(NSUInteger i, BOOL *stop) {
        (void)stop;
        if (prev != NSNotFound && prev != i + 1) [thread addObject:@{@"gap": @YES}];
        NSMutableDictionary *m = [msgs[i] mutableCopy];
        if ([hits containsIndex:i]) m[@"hit"] = @YES;
        [thread addObject:m];
        prev = i;
    }];

    void (^finish)(NSString *, NSString *) = ^(NSString *text, NSString *badge) {
        [self recordImmediateKronosReply:text actionBadge:nil replyBody:replyBody replyExtra:replyExtra convMsgs:convMsgs];
        [self addWhatsAppThread:thread toStack:replyExtra];
        NSMutableDictionary *saved = [convMsgs.lastObject mutableCopy];
        saved[@"waThread"] = thread;
        saved[@"actionBadge"] = badge;
        convMsgs[convMsgs.count - 1] = saved;
        [self addActionPillWithTitle:badge previewFile:nil toStack:replyExtra];
        [KronosConversationStore saveConversations:self.conversations];
        [self scrollToBottomAnimated:YES];
    };
    if (intent != KronosMsgIntentThreadOverview) {
        finish(reply, @"Verified WhatsApp Message");
        return YES;
    }

    // The Mac model can weigh the thread; the phone's model misquoted it, so it is not used here.
    NSMutableArray<NSString *> *timeline = [NSMutableArray array];
    for (NSUInteger i = MIN((NSUInteger)30, msgs.count); i > 0; i--) {
        NSDictionary *m = msgs[i - 1];
        [timeline addObject:[NSString stringWithFormat:@"[%@] %@: %@", m[@"t"], m[@"who"], m[@"text"]]];
    }
    NSString *prompt = [NSString stringWithFormat:@"WhatsApp messages between the user (\"You\") and %@, oldest first:\n%@\n\nQuestion from the user: %@\n\nAnswer directly in one or two sentences, speaking to the user, using only these messages. Pay most attention to what %@ wrote. Mention the message and time that settle it. If they do not settle it, say so.", contactName, [timeline componentsJoinedByString:@"\n"], question, contactName];
    replyBody.text = @"Reading the chat…";
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        NSString *answer = kronosTryGenerateChatViaLocalMac14B(@[@{@"role": @"user", @"content": prompt}], 160, 0.0f);
        answer = [[kronosStripThinkBlocks(answer ?: @"") stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]] copy];
        dispatch_async(dispatch_get_main_queue(), ^{
            if (answer.length > 0) finish(answer, @"Read from your WhatsApp");
            else finish(reply, @"Verified WhatsApp Message");
        });
    });
    return YES;
}

/// Chat bubbles as in WhatsApp: yours on the right in green, theirs on the left, time under the text.
- (void)addWhatsAppThread:(NSArray<NSDictionary *> *)thread toStack:(UIStackView *)stack {
    UILabel *title = [[UILabel alloc] init];
    NSString *contact = @"";
    for (NSDictionary *m in thread) {
        if (![m[@"who"] isEqualToString:@"You"]) { contact = m[@"who"]; break; }
    }
    title.text = contact.length > 0 ? [NSString stringWithFormat:@"WhatsApp · %@", contact] : @"WhatsApp";
    title.font = [UIFont systemFontOfSize:12.0 weight:UIFontWeightSemibold];
    title.textColor = kronosMutedColor();
    [stack addArrangedSubview:title];

    UIColor *mineColor = kronosDynamicColor(0xD9FDD3, 1.0, 0x005C4B, 1.0);
    UIColor *theirColor = kronosDynamicColor(0xF0F0EE, 1.0, 0x202C33, 1.0);
    UIStackView *list = [[UIStackView alloc] init];
    list.axis = UILayoutConstraintAxisVertical;
    list.spacing = 5.0;
    for (NSDictionary *m in thread) {
        if ([m[@"gap"] boolValue]) {
            UILabel *gap = [[UILabel alloc] init];
            gap.text = @"···";
            gap.textAlignment = NSTextAlignmentCenter;
            gap.textColor = kronosMutedColor();
            gap.font = [UIFont systemFontOfSize:13.0 weight:UIFontWeightBold];
            [list addArrangedSubview:gap];
            continue;
        }
        BOOL mine = [m[@"who"] isEqualToString:@"You"];
        UIView *row = [[UIView alloc] init];
        UIView *bubble = [[UIView alloc] init];
        bubble.translatesAutoresizingMaskIntoConstraints = NO;
        bubble.backgroundColor = mine ? mineColor : theirColor;
        bubble.layer.cornerRadius = 10.0;
        if ([m[@"hit"] boolValue]) {
            bubble.layer.borderWidth = 1.5;
            bubble.layer.borderColor = kronosAccentColor().CGColor;
        }
        [row addSubview:bubble];

        UILabel *text = [[UILabel alloc] init];
        text.translatesAutoresizingMaskIntoConstraints = NO;
        text.numberOfLines = 0;
        text.text = m[@"text"];
        text.font = [UIFont systemFontOfSize:15.0];
        text.textColor = kronosTextColor();
        [bubble addSubview:text];

        UILabel *time = [[UILabel alloc] init];
        time.translatesAutoresizingMaskIntoConstraints = NO;
        time.text = m[@"t"];
        time.font = [UIFont systemFontOfSize:11.0];
        time.textColor = [kronosTextColor() colorWithAlphaComponent:0.6];
        [bubble addSubview:time];

        [NSLayoutConstraint activateConstraints:@[
            [bubble.topAnchor constraintEqualToAnchor:row.topAnchor],
            [bubble.bottomAnchor constraintEqualToAnchor:row.bottomAnchor],
            mine ? [bubble.trailingAnchor constraintEqualToAnchor:row.trailingAnchor]
                 : [bubble.leadingAnchor constraintEqualToAnchor:row.leadingAnchor],
            mine ? [bubble.leadingAnchor constraintGreaterThanOrEqualToAnchor:row.leadingAnchor constant:48.0]
                 : [bubble.trailingAnchor constraintLessThanOrEqualToAnchor:row.trailingAnchor constant:-48.0],
            [text.topAnchor constraintEqualToAnchor:bubble.topAnchor constant:7.0],
            [text.leadingAnchor constraintEqualToAnchor:bubble.leadingAnchor constant:10.0],
            [text.trailingAnchor constraintEqualToAnchor:bubble.trailingAnchor constant:-10.0],
            [time.topAnchor constraintEqualToAnchor:text.bottomAnchor constant:2.0],
            [time.leadingAnchor constraintGreaterThanOrEqualToAnchor:bubble.leadingAnchor constant:10.0],
            [time.trailingAnchor constraintEqualToAnchor:bubble.trailingAnchor constant:-10.0],
            [time.bottomAnchor constraintEqualToAnchor:bubble.bottomAnchor constant:-5.0],
        ]];
        [list addArrangedSubview:row];
    }
    [stack addArrangedSubview:list];
}

- (void)answerApp:(NSString *)app ask:(KronosAppAsk)ask fallbackFind:(BOOL)fallbackFind replyBody:(UILabel *)replyBody replyExtra:(UIStackView *)replyExtra convMsgs:(NSMutableArray *)convMsgs {
    replyBody.text = @"Checking your apps…";
    [self macRequest:@"GET" path:@"/v1/device/phone_apps" body:nil timeout:(fallbackFind ? 6.0 : 30.0) completion:^(NSInteger status, NSDictionary *json) {
        NSArray *apps = [json[@"apps"] isKindOfClass:[NSArray class]] ? json[@"apps"] : nil;
        if (status != 200 || apps.count == 0) {
            if (fallbackFind) {
                [self answerFindRequest:app replyBody:replyBody replyExtra:replyExtra convMsgs:convMsgs];
                return;
            }
            [self answerAppWithoutMac:app ask:ask error:json[@"error"] replyBody:replyBody replyExtra:replyExtra convMsgs:convMsgs];
            return;
        }
        NSMutableArray<NSDictionary *> *hits = [NSMutableArray array];
        for (NSDictionary *a in apps) {
            if ([a isKindOfClass:[NSDictionary class]] && kronosAppNameMatches(a[@"name"] ?: @"", app)) [hits addObject:a];
        }
        NSMutableArray<NSString *> *names = [NSMutableArray array];
        for (NSDictionary *a in hits) [names addObject:a[@"name"]];
        NSString *joined = [names componentsJoinedByString:@", "];
        NSString *reply;
        if (ask == KronosAppAskCount) {
            if (app.length == 0) reply = [NSString stringWithFormat:@"You have %lu apps on your iPhone, including Apple's built-in ones.", (unsigned long)apps.count];
            else if (hits.count == 0) reply = [NSString stringWithFormat:@"None. I checked the %lu apps on your iPhone and none is named \"%@\".", (unsigned long)apps.count, app];
            else reply = [NSString stringWithFormat:@"%lu %@ \"%@\": %@.", (unsigned long)hits.count, hits.count == 1 ? @"app matches" : @"apps match", app, joined];
        } else if (hits.count == 0) {
            if (fallbackFind) {
                [self answerFindRequest:app replyBody:replyBody replyExtra:replyExtra convMsgs:convMsgs];
                return;
            }
            reply = [NSString stringWithFormat:@"No, I checked the %lu apps on your iPhone and none is named \"%@\".", (unsigned long)apps.count, app];
        } else if (ask == KronosAppAskOpen && hits.count == 1) {
            [self openPhoneApp:hits.firstObject replyBody:replyBody replyExtra:replyExtra convMsgs:convMsgs];
            return;
        } else if (hits.count == 1) {
            reply = [NSString stringWithFormat:@"Yes, %@ is installed on your iPhone.", joined];
        } else {
            reply = [NSString stringWithFormat:@"Yes, these match \"%@\": %@.", app, joined];
        }
        [self recordImmediateKronosReply:reply actionBadge:nil replyBody:replyBody replyExtra:replyExtra convMsgs:convMsgs];
        for (NSDictionary *a in hits) {
            if (hits.count > 8) break;
            [self addActionPillWithTitle:[NSString stringWithFormat:@"Open %@", a[@"name"]] symbol:@"arrow.up.forward.app" action:@selector(openPhoneAppTapped:) toStack:replyExtra];
            UIView *pill = replyExtra.arrangedSubviews.lastObject;
            pill.accessibilityIdentifier = a[@"bundle"];
            pill.accessibilityLabel = a[@"name"];
        }
    }];
}

- (void)openPhoneApp:(NSDictionary *)app replyBody:(UILabel *)replyBody replyExtra:(UIStackView *)replyExtra convMsgs:(NSMutableArray *)convMsgs {
    replyBody.text = [NSString stringWithFormat:@"Opening %@…", app[@"name"]];
    [self macRequest:@"POST" path:@"/v1/device/phone_apps/open" body:@{@"bundle": app[@"bundle"] ?: @""} timeout:30.0 completion:^(NSInteger status, NSDictionary *json) {
        NSString *reply = status == 200 ? [NSString stringWithFormat:@"Opened %@.", app[@"name"]]
            : [NSString stringWithFormat:@"I couldn't open %@. %@", app[@"name"], json[@"error"] ?: @"Your Mac did not respond."];
        [self recordImmediateKronosReply:reply actionBadge:nil replyBody:replyBody replyExtra:replyExtra convMsgs:convMsgs];
    }];
}

- (void)openPhoneAppTapped:(UIButton *)sender {
    NSString *name = sender.accessibilityLabel ?: @"the app";
    [self macRequest:@"POST" path:@"/v1/device/phone_apps/open" body:@{@"bundle": sender.accessibilityIdentifier ?: @""} timeout:30.0 completion:^(NSInteger status, NSDictionary *json) {
        (void)json;
        if (status != 200) kronosShowToast(self.view, [NSString stringWithFormat:@"Could not open %@", name]);
    }];
}

/// Without the Mac, iOS only answers for the URL schemes this app declares.
- (void)answerAppWithoutMac:(NSString *)app ask:(KronosAppAsk)ask error:(id)error replyBody:(UILabel *)replyBody replyExtra:(UIStackView *)replyExtra convMsgs:(NSMutableArray *)convMsgs {
    NSDictionary *schemes = @{@"whatsapp": @"whatsapp", @"telegram": @"tg", @"slack": @"slack", @"youtube": @"youtube", @"instagram": @"instagram", @"twitter": @"twitter", @"github": @"github", @"linkedin": @"linkedin", @"gmail": @"googlegmail"};
    NSString *scheme = schemes[[app lowercaseString]];
    NSURL *url = scheme != nil ? [NSURL URLWithString:[scheme stringByAppendingString:@"://"]] : nil;
    NSString *reply;
    if (url != nil && ask != KronosAppAskCount) {
        BOOL has = [[UIApplication sharedApplication] canOpenURL:url];
        if (has && ask == KronosAppAskOpen) {
            [[UIApplication sharedApplication] openURL:url options:@{} completionHandler:nil];
            reply = [NSString stringWithFormat:@"Opened %@.", app];
        } else {
            reply = has ? [NSString stringWithFormat:@"Yes, %@ is installed on your iPhone.", app] : [NSString stringWithFormat:@"No, %@ is not installed on your iPhone.", app];
        }
    } else {
        NSString *why = [error isKindOfClass:[NSString class]] ? [NSString stringWithFormat:@" Your Mac reported: %@.", error] : @"";
        reply = [NSString stringWithFormat:@"I can't check your apps right now. iOS hides the app list from Kronos, so it asks your Mac, and that didn't work.%@ Check the bridge is running and your iPhone is on the same Wi-Fi, or connect the cable.", why];
    }
    [self recordImmediateKronosReply:reply actionBadge:nil replyBody:replyBody replyExtra:replyExtra convMsgs:convMsgs];
}

- (void)macRequest:(NSString *)method path:(NSString *)path body:(NSDictionary *)body completion:(void (^)(NSInteger status, NSDictionary *json))completion {
    [self macRequest:method path:path body:body timeout:8.0 completion:completion];
}

- (void)macRequest:(NSString *)method path:(NSString *)path body:(NSDictionary *)body timeout:(NSTimeInterval)timeout completion:(void (^)(NSInteger status, NSDictionary *json))completion {
    KronosMobileSettings *settings = [KronosMobileSettings sharedSettings];
    NSString *base = [settings.macServerURL stringByTrimmingCharactersInSet:[NSCharacterSet characterSetWithCharactersInString:@"/ "]];
    NSURL *url = settings.macSyncEnabled ? [NSURL URLWithString:[base stringByAppendingString:path]] : nil;
    if (url == nil) { completion(0, nil); return; }
    NSMutableURLRequest *req = [NSMutableURLRequest requestWithURL:url cachePolicy:NSURLRequestReloadIgnoringLocalCacheData timeoutInterval:timeout];
    req.HTTPMethod = method;
    kronosApplyMacAuth(req);
    if (body != nil) {
        [req setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
        req.HTTPBody = [NSJSONSerialization dataWithJSONObject:body options:0 error:nil];
    }
    [[[NSURLSession sharedSession] dataTaskWithRequest:req completionHandler:^(NSData *data, NSURLResponse *response, NSError *error) {
        NSInteger status = (error == nil && [response isKindOfClass:[NSHTTPURLResponse class]]) ? [(NSHTTPURLResponse *)response statusCode] : 0;
        id json = data.length > 0 ? [NSJSONSerialization JSONObjectWithData:data options:0 error:nil] : nil;
        dispatch_async(dispatch_get_main_queue(), ^{
            completion(status, [json isKindOfClass:[NSDictionary class]] ? json : nil);
        });
    }] resume];
}

- (void)startMacTask:(NSString *)task replyBody:(UILabel *)replyBody replyExtra:(UIStackView *)replyExtra convMsgs:(NSMutableArray *)convMsgs {
    replyBody.text = @"Sending the task to your Mac…";
    [self macRequest:@"POST" path:@"/v1/device/tasks" body:@{@"prompt": task} completion:^(NSInteger status, NSDictionary *json) {
        if (status != 200 || json[@"id"] == nil) {
            NSString *why = [json[@"error"] isKindOfClass:[NSString class]] ? json[@"error"]
                : [NSString stringWithFormat:@"Could not reach your Mac at %@. Check it is awake and on the same Wi-Fi.", [KronosMobileSettings sharedSettings].macServerURL];
            [self recordImmediateKronosReply:why actionBadge:nil replyBody:replyBody replyExtra:replyExtra convMsgs:convMsgs];
            return;
        }
        [self followMacTask:json[@"id"] replyBody:replyBody replyExtra:replyExtra convMsgs:convMsgs];
    }];
}

- (void)followMacTask:(NSNumber *)taskId replyBody:(UILabel *)replyBody replyExtra:(UIStackView *)replyExtra convMsgs:(NSMutableArray *)convMsgs {
    [self addActionPillWithTitle:@"Cancel task" symbol:@"xmark.circle" action:@selector(cancelMacTaskTapped:) toStack:replyExtra];
    UIView *cancelPill = replyExtra.arrangedSubviews.lastObject;
    cancelPill.accessibilityIdentifier = taskId.stringValue;
    [self pollMacTask:taskId attempt:0 cancelPill:cancelPill replyBody:replyBody replyExtra:replyExtra convMsgs:convMsgs];
}

- (void)pollMacTask:(NSNumber *)taskId attempt:(NSUInteger)attempt cancelPill:(UIView *)cancelPill replyBody:(UILabel *)replyBody replyExtra:(UIStackView *)replyExtra convMsgs:(NSMutableArray *)convMsgs {
    NSString *path = [NSString stringWithFormat:@"/v1/device/tasks?id=%@", taskId];
    [self macRequest:@"GET" path:path body:nil completion:^(NSInteger status, NSDictionary *json) {
        NSArray *tasks = [json[@"tasks"] isKindOfClass:[NSArray class]] ? json[@"tasks"] : @[];
        NSDictionary *t = [tasks.firstObject isKindOfClass:[NSDictionary class]] ? tasks.firstObject : nil;
        NSString *state = t[@"state"] ?: @"";
        NSString *final = nil;
        if ([state isEqualToString:@"done"]) {
            NSString *text = t[@"text"] ?: @"";
            NSRange note = [text rangeOfString:@"\n[No tool calls"];
            if (note.location != NSNotFound) text = [text substringToIndex:note.location];
            NSArray *files = t[@"deliverables"];
            final = [NSString stringWithFormat:@"Finished on your Mac.\n\n%@", text];
            if (files.count > 0) final = [final stringByAppendingFormat:@"\n\nFiles: %@", [files componentsJoinedByString:@", "]];
        } else if ([state isEqualToString:@"denied"]) {
            final = @"The task did not run. It was denied on your Mac, or nobody approved it in time.";
        } else if ([state isEqualToString:@"cancelled"]) {
            final = @"Task cancelled.";
        } else if ([state isEqualToString:@"failed"]) {
            NSString *d = t[@"detail"];
            final = [NSString stringWithFormat:@"The task failed on your Mac. %@", d ?: @""];
        } else if (t == nil && status == 200) {
            final = @"Your Mac no longer has this task. The bridge may have restarted.";
        } else if (attempt >= 1200) {
            final = @"The task is still running on your Mac. Ask \"what is my Mac doing\" to check on it.";
        }
        if (final != nil) {
            [cancelPill removeFromSuperview];
            [self recordImmediateKronosReply:final actionBadge:nil replyBody:replyBody replyExtra:replyExtra convMsgs:convMsgs];
            return;
        }
        if (t == nil) replyBody.text = @"Waiting to reconnect to your Mac…";
        else if ([state isEqualToString:@"awaiting_approval"]) replyBody.text = @"Waiting for you to approve the task on your Mac.";
        else {
            NSString *d = t[@"detail"];
            replyBody.text = [NSString stringWithFormat:@"Running on your Mac. %@", d ?: @""];
        }
        dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(3 * NSEC_PER_SEC)), dispatch_get_main_queue(), ^{
            [self pollMacTask:taskId attempt:attempt + 1 cancelPill:cancelPill replyBody:replyBody replyExtra:replyExtra convMsgs:convMsgs];
        });
    }];
}

- (void)cancelMacTaskTapped:(UIButton *)sender {
    NSInteger taskId = sender.accessibilityIdentifier.integerValue;
    [self macRequest:@"POST" path:@"/v1/device/tasks/cancel" body:@{@"id": @(taskId)} completion:^(NSInteger status, NSDictionary *json) {
        (void)json;
        kronosShowToast(self.view, status == 200 ? @"Cancelling" : @"Could not reach your Mac");
    }];
}

- (void)answerMacTaskStatusWithReplyBody:(UILabel *)replyBody replyExtra:(UIStackView *)replyExtra convMsgs:(NSMutableArray *)convMsgs {
    [self macRequest:@"GET" path:@"/v1/device/tasks" body:nil completion:^(NSInteger status, NSDictionary *json) {
        NSArray *tasks = [json[@"tasks"] isKindOfClass:[NSArray class]] ? json[@"tasks"] : nil;
        if (status != 200 || tasks == nil) {
            [self recordImmediateKronosReply:@"Could not reach your Mac." actionBadge:nil replyBody:replyBody replyExtra:replyExtra convMsgs:convMsgs];
            return;
        }
        NSDictionary *t = tasks.firstObject;
        if (t == nil) {
            [self recordImmediateKronosReply:@"Your Mac has no tasks from this iPhone." actionBadge:nil replyBody:replyBody replyExtra:replyExtra convMsgs:convMsgs];
            return;
        }
        [self followMacTask:t[@"id"] replyBody:replyBody replyExtra:replyExtra convMsgs:convMsgs];
    }];
}

- (void)openSavedMediaTapped:(UIButton *)sender {
    NSString *path = sender.accessibilityIdentifier;
    if (![[NSFileManager defaultManager] fileExistsAtPath:path ?: @""]) return;
    KronosFilePreviewViewController *pvc = [[KronosFilePreviewViewController alloc] init];
    pvc.filePath = path;
    UINavigationController *nav = [[UINavigationController alloc] initWithRootViewController:pvc];
    [self presentViewController:nav animated:YES completion:nil];
}

- (void)openAppSettingsTapped {
    [[UIApplication sharedApplication] openURL:[NSURL URLWithString:UIApplicationOpenSettingsURLString] options:@{} completionHandler:nil];
}

/// A pill that runs `action` on self (pickers, retries).
- (void)addActionPillWithTitle:(NSString *)title symbol:(NSString *)symbol action:(SEL)action toStack:(UIStackView *)stack {
    if (title.length == 0 || stack == nil) return;
    UIButton *pill = [KronosTapButton tapButton];
    pill.translatesAutoresizingMaskIntoConstraints = NO;
    [pill setImage:[UIImage systemImageNamed:symbol ?: @"arrow.right.circle"] forState:UIControlStateNormal];
    [pill setTitle:[NSString stringWithFormat:@"  %@", title] forState:UIControlStateNormal];
    pill.tintColor = kronosAccentColor();
    [pill setTitleColor:kronosAccentColor() forState:UIControlStateNormal];
    pill.titleLabel.font = [UIFont systemFontOfSize:13.0 weight:UIFontWeightSemibold];
    pill.contentHorizontalAlignment = UIControlContentHorizontalAlignmentLeading;
    pill.backgroundColor = [kronosAccentColor() colorWithAlphaComponent:0.12];
    pill.layer.cornerRadius = 10.0;
    [pill.heightAnchor constraintEqualToConstant:36.0].active = YES;
    [pill addTarget:self action:action forControlEvents:UIControlEventTouchUpInside];
    [stack addArrangedSubview:pill];
}

/// Workspace files and this conversation's attachment names matching `term` (all of them for "").
- (NSArray<NSString *> *)searchReadableFilesForTerm:(NSString *)term {
    NSMutableArray<NSString *> *names = [NSMutableArray array];
    for (NSString *f in [[NSFileManager defaultManager] contentsOfDirectoryAtPath:kronosWorkspaceDirectory() error:nil] ?: @[]) {
        if (![f hasPrefix:@"."]) [names addObject:f];
    }
    for (NSDictionary *m in self.activeConversation[@"messages"] ?: @[]) {
        for (NSString *a in m[@"attachments"] ?: @[]) {
            if (![names containsObject:a]) [names addObject:a];
        }
    }
    if (term.length == 0) return names;
    NSMutableArray<NSString *> *hits = [NSMutableArray array];
    for (NSString *n in names) {
        if (kronosFileNameMatchesTerm(n, term)) [hits addObject:n];
    }
    return hits;
}

- (void)pickDocumentTapped {
    UIDocumentPickerViewController *doc = [[UIDocumentPickerViewController alloc] initForOpeningContentTypes:@[UTTypeItem] asCopy:YES];
    doc.allowsMultipleSelection = YES;
    doc.delegate = self;
    [self presentViewController:doc animated:YES completion:nil];
}

- (void)pickPhotoTapped {
    PHPickerConfiguration *cfg = [[PHPickerConfiguration alloc] init];
    cfg.selectionLimit = 4;
    cfg.filter = [PHPickerFilter imagesFilter];
    PHPickerViewController *picker = [[PHPickerViewController alloc] initWithConfiguration:cfg];
    picker.delegate = self;
    [self presentViewController:picker animated:YES completion:nil];
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
    UIButton *btn = [KronosTapButton tapButton];
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
    UIButton *shareBtn = [KronosTapButton tapButton];
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

    UIButton *feedBtn = [KronosTapButton tapButton];
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
    if (body.text.length == 0) return;
    [UIPasteboard generalPasteboard].string = body.text;
    [[[UINotificationFeedbackGenerator alloc] init] notificationOccurred:UINotificationFeedbackTypeSuccess];
    if ([sender isKindOfClass:[KronosTapButton class]]) {
        [(KronosTapButton *)sender flashSymbol:@"checkmark" tint:kronosAccentColor()];
    }
    kronosShowToast(self.view, @"Copied");
}

/// Speaks the bubble, or stops if this bubble is the one speaking; the icon pulses while audio plays.
- (void)speakBubbleTapped:(UIButton *)sender {
    UIView *cur = sender;
    while (cur != nil && [cur viewWithTag:3301] == nil) cur = cur.superview;
    UILabel *body = (UILabel *)[cur viewWithTag:3301];
    KronosTapButton *btn = [sender isKindOfClass:[KronosTapButton class]] ? (KronosTapButton *)sender : nil;
    if (self.activeSpeakButton == btn && btn != nil) {
        [self.voiceController stopSpeaking];
        return;
    }
    if (body.text.length == 0) return;
    [self clearSpeakingState];
    self.activeSpeakButton = btn;
    [btn setImage:[UIImage systemImageNamed:@"speaker.wave.2.fill"] forState:UIControlStateNormal];
    btn.tintColor = kronosAccentColor();
    btn.pulsing = YES;
    __weak typeof(self) weakSelf = self;
    self.voiceController.onSpeechFinished = ^{ [weakSelf clearSpeakingState]; };
    [self.voiceController speakText:body.text];
}

- (void)clearSpeakingState {
    KronosTapButton *btn = self.activeSpeakButton;
    if (btn != nil) {
        btn.pulsing = NO;
        [btn setImage:[UIImage systemImageNamed:@"speaker.wave.2"] forState:UIControlStateNormal];
        btn.tintColor = kronosMutedColor();
    }
    self.activeSpeakButton = nil;
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

    // 0h. App questions: iOS hides the app list, so the paired Mac answers. Never the model.
    BOOL afterAppTurn = self.lastTurnWasAppAsk;
    self.lastTurnWasAppAsk = NO;
    NSString *askedApp = nil;
    KronosAppAsk appAsk = KronosAppAskHas;
    if (turnAttachments.count == 0 && !awaitingWhatsAppDraftAnswer) {
        BOOL isAppAsk = kronosParseHasAppIntent(effectiveQuery, &askedApp) || kronosParseAppAsk(effectiveQuery, afterAppTurn, &askedApp, &appAsk);
        if (!isAppAsk && afterAppTurn && self.lastAskedApp.length > 0 &&
            ([trimmedLower isEqualToString:@"open it"] || [trimmedLower isEqualToString:@"open that"] || [trimmedLower isEqualToString:@"open that app"] || [trimmedLower isEqualToString:@"open it please"])) {
            askedApp = self.lastAskedApp;
            appAsk = KronosAppAskOpen;
            isAppAsk = YES;
        }
        if (isAppAsk && !kronosParseWhatsAppDraftIntent(userText, NULL, NULL, NULL, NULL, NULL)) {
            self.lastTurnWasAppAsk = YES;
            self.lastAskedApp = askedApp;
            [self answerApp:askedApp ask:appAsk fallbackFind:NO replyBody:replyBody replyExtra:replyExtra convMsgs:convMsgs];
            return;
        }
    }

    // 0m. "On my Mac, ...": run an agent task on the Mac after approval there.
    NSString *macTask = nil;
    if (turnAttachments.count == 0 && !awaitingWhatsAppDraftAnswer && kronosIsMacTaskStatusQuery(effectiveQuery)) {
        [self answerMacTaskStatusWithReplyBody:replyBody replyExtra:replyExtra convMsgs:convMsgs];
        return;
    }
    if (turnAttachments.count == 0 && !awaitingWhatsAppDraftAnswer && kronosParseMacTaskIntent(effectiveQuery, &macTask)) {
        [self startMacTask:macTask replyBody:replyBody replyExtra:replyExtra convMsgs:convMsgs];
        return;
    }

    // 0a. "Last screenshot I took": read Photos directly. No model call.
    NSInteger mediaKind = 0;
    BOOL mediaCount = NO;
    if (turnAttachments.count == 0 && !awaitingWhatsAppDraftAnswer && kronosParseLatestMediaIntent(effectiveQuery, &mediaKind, &mediaCount)
        && !kronosParseWhatsAppDraftIntent(userText, NULL, NULL, NULL, NULL, NULL) && !kronosParseLinkedInPostIntent(userText, NULL, NULL)) {
        [self answerLatestMediaKind:mediaKind countOnly:mediaCount replyBody:replyBody replyExtra:replyExtra convMsgs:convMsgs];
        return;
    }

    // 0. "Find / open my X": search what Kronos can read, or offer the pickers. No model call.
    NSString *findTerm = nil;
    if (turnAttachments.count == 0 && !awaitingWhatsAppDraftAnswer && kronosParseFindFileIntent(effectiveQuery, &findTerm)
        && !kronosParseWhatsAppDraftIntent(userText, NULL, NULL, NULL, NULL, NULL) && !kronosParseLinkedInPostIntent(userText, NULL, NULL)) {
        if ([trimmedLower hasPrefix:@"open "] || [trimmedLower hasPrefix:@"launch "]) {
            self.pendingFindTerm = findTerm;
            [self answerApp:findTerm ask:KronosAppAskOpen fallbackFind:YES replyBody:replyBody replyExtra:replyExtra convMsgs:convMsgs];
            return;
        }
        [self answerFindRequest:findTerm replyBody:replyBody replyExtra:replyExtra convMsgs:convMsgs];
        return;
    }

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
        // Mac model resolves the pronouns; rules are the fallback.
        NSString *waLow = [[effectiveQuery stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]] lowercaseString];
        if (!waIsTopic && kronosIsTellPersonThat(waLow)) {
            replyBody.text = @"Writing the message…";
            NSString *prompt = [NSString stringWithFormat:@"The user is talking to their assistant, Kronos, and said: \"%@\"\n\nWrite the WhatsApp message the user would send to %@, in the user's voice. \"I\" means the user. \"you\" in the request means Kronos, so write \"Kronos\". \"she\", \"he\" or \"her\" for %@ become \"you\" or \"your\". \"the message above\" means the user's last message to %@. Keep the user's tone and every point. Output only the message.", effectiveQuery, waRecipient, waRecipient, waRecipient];
            NSString *fallback = waExplicitMsg;
            dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
                NSString *rewritten = kronosTryGenerateChatViaLocalMac14B(@[@{@"role": @"user", @"content": prompt}], 200, 0.0f);
                rewritten = [kronosStripThinkBlocks(rewritten ?: @"") stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
                dispatch_async(dispatch_get_main_queue(), ^{
                    [self completeWhatsAppDraftForRecipient:waRecipient
                                                rawUserSpec:(rewritten.length > 0 ? rewritten : fallback)
                                              isInstruction:NO
                                                  replyBody:replyBody
                                                 replyExtra:replyExtra
                                                   convMsgs:convMsgs];
                });
            });
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
                    if ([self answerThreadQuestion:effectiveQuery fromContext:verifiedContextOrDirectReply replyBody:replyBody replyExtra:replyExtra convMsgs:convMsgs]) {
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
            BOOL thinking = kronosHasOpenThinkBlock(snap);
            NSString *cleanSnap = thinking ? @"Thinking…" : kronosSanitizeFirstPersonReply(kronosStripThinkBlocks(snap));
            dispatch_async(dispatch_get_main_queue(), ^{
                replyBody.attributedText = kronosFormatMessageText(cleanSnap, NO);
                replyBody.textColor = thinking ? kronosMutedColor() : kronosTextColor();
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

        finalText = kronosVerifyAndRepairIdentityReply(kronosStripThinkBlocks(finalText), userText);
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
