// actor_pose_publisher.cc
#include <ignition/gazebo/System.hh>
#include <ignition/gazebo/Actor.hh>
#include <ignition/gazebo/EntityComponentManager.hh>
#include <ignition/gazebo/components/Name.hh>
#include <ignition/transport/Node.hh>
#include <ignition/msgs/pose.pb.h>
#include <ignition/msgs/Utility.hh>
#include <chrono>

using namespace ignition;
using namespace ignition::gazebo;

class ActorPosePublisher final
  : public System, public ISystemConfigure, public ISystemPreUpdate {
 public:
  void Configure(const Entity&, const std::shared_ptr<const sdf::Element> &sdf,
                 EntityComponentManager&, EventManager&) override {
    if (sdf && sdf->HasElement("actor_name"))
      actorName_ = sdf->Get<std::string>("actor_name");
    if (sdf && sdf->HasElement("topic"))
      topic_ = sdf->Get<std::string>("topic");
    if (sdf && sdf->HasElement("rate_hz"))
      rateHz_ = sdf->Get<double>("rate_hz");

    pub_ = node_.Advertise<ignition::msgs::Pose>(topic_);
  }

  void PreUpdate(const UpdateInfo &info, EntityComponentManager &ecm) override {
    if (info.paused) return;

    // Resolve actor entity by name (one-time)
    if (actorEntity_ == kNullEntity) {
      ecm.Each<components::Name>(
        [&](const Entity &e, const components::Name *n)->bool {
          if (n->Data() == actorName_) {
            Actor a(e);
            if (a.Valid(ecm)) { actorEntity_ = e; return false; }
          }
          return true;
        });
      if (actorEntity_ == kNullEntity) return; // not found yet
    }

    // 30 Hz throttle (sim time)
    const auto now = std::chrono::nanoseconds(info.simTime.count());
    const auto period = std::chrono::nanoseconds(
        static_cast<int64_t>(1e9 / rateHz_));
    if (lastEmit_ != std::chrono::nanoseconds::zero()
        && now - lastEmit_ < period) return;

    Actor a(actorEntity_);
    if (!a.Valid(ecm)) return;

    const auto pose = a.WorldPose(ecm); // ignition::math::Pose3d
    ignition::msgs::Pose msg;
    ignition::msgs::Set(msg.mutable_position(),    pose.Pos());
    ignition::msgs::Set(msg.mutable_orientation(), pose.Rot());
    pub_.Publish(msg);
    lastEmit_ = now;
  }

 private:
  std::string actorName_{"linear_actor"};
  std::string topic_{"/linear_actor/pose"};
  double rateHz_{30.0};

  Entity actorEntity_{kNullEntity};
  ignition::transport::Node node_;
  ignition::transport::Node::Publisher pub_;
  std::chrono::nanoseconds lastEmit_{};
};

IGNITION_ADD_PLUGIN(ActorPosePublisher,
  ignition::gazebo::System,
  ignition::gazebo::ISystemConfigure,
  ignition::gazebo::ISystemPreUpdate)

IGNITION_ADD_PLUGIN_ALIAS(ActorPosePublisher, "ActorPosePublisher")
